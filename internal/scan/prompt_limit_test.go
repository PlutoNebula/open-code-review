// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package scan

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/alibaba/open-code-review/internal/config/template"
	"github.com/alibaba/open-code-review/internal/llm"
	"github.com/alibaba/open-code-review/internal/llmloop"
	"github.com/alibaba/open-code-review/internal/model"
	"github.com/alibaba/open-code-review/internal/session"
	"github.com/alibaba/open-code-review/internal/tool"
)

func TestRun_RenderedPromptLimit(t *testing.T) {
	for _, tc := range []struct {
		name       string
		contents   []string
		batchSize  int
		wantFailed int64
	}{
		{"single rejected file", []string{strings.Repeat("word ", 60)}, 1, 1},
		{"all rejected files", []string{strings.Repeat("word ", 60), strings.Repeat("word ", 60)}, 2, 2},
		{"mixed batch", []string{strings.Repeat("word ", 60), "package main\n"}, 2, 1},
		{"later batch continues", []string{strings.Repeat("word ", 60), "package main\n"}, 1, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			testHome := t.TempDir()
			t.Setenv("HOME", testHome)
			t.Setenv("USERPROFILE", testHome)
			repo := t.TempDir()
			items := make([]model.ScanItem, len(tc.contents))
			for i, content := range tc.contents {
				name := string(rune('a'+i)) + ".go"
				if err := os.WriteFile(filepath.Join(repo, name), []byte(content), 0o600); err != nil {
					t.Fatal(err)
				}
				items[i] = model.ScanItem{Path: name, Content: content}
			}
			client := &fakeBudgetClient{perCallTokens: 1}
			tpl := budgetTestTemplate()
			tpl.MaxTokens = 100
			tpl.BatchStrategy = string(BatchByLanguage)
			tpl.BatchSize = tc.batchSize
			tpl.MainTask.Messages[0].Content = strings.Repeat("rule ", 40)
			a := NewAgent(Args{
				RepoDir: repo, Template: tpl, LLMClient: client, Model: "test-model",
				Tools: tool.NewRegistry(), CommentCollector: tool.NewCommentCollector(),
				MaxConcurrency: 1, SkipPlan: true, SkipDedup: true, SkipSummary: true,
				Session: session.New(repo, "main", "test-model", session.SessionOptions{ReviewMode: session.ReviewModeFullScan}),
			})
			for _, decision := range a.selectScanItems(items) {
				if decision.reason != model.ExcludeNone {
					t.Fatalf("fixture did not pass raw-file selection: %+v", decision)
				}
			}
			rejected := make(map[string]bool, len(items))
			for _, item := range items {
				count := llmloop.CountMessagesTokens(a.renderMessages(item, "", ""))
				rejected[item.Path] = count > llmloop.PromptTokenLimit(tpl.MaxTokens)
				if rejected[item.Path] != strings.Contains(item.Content, "word") {
					t.Fatalf("unexpected rendered fixture size for %s: %d", item.Path, count)
				}
			}
			_, err := a.Run(t.Context())
			allFailed := tc.wantFailed == int64(len(items))
			if (err != nil) != allFailed {
				t.Fatalf("Run error = %v, all failed = %v", err, allFailed)
			}
			if allFailed && !strings.Contains(err.Error(), "all ") {
				t.Fatalf("expected all-files-failed error, got %v", err)
			}
			if got := atomic.LoadInt64(&a.subtaskFailed); got != tc.wantFailed {
				t.Errorf("failed = %d, want %d", got, tc.wantFailed)
			}
			wantCalls := int64(len(items)) - tc.wantFailed
			if got := atomic.LoadInt64(&client.calls); got != wantCalls {
				t.Errorf("main review calls = %d, want %d", got, wantCalls)
			}
			warningPaths := map[string]bool{}
			for _, w := range a.Warnings() {
				if w.Type == "scan_subtask_error" {
					if w.Message == "" {
						t.Error("incomplete review warning has no reason")
					}
					warningPaths[w.File] = true
				}
			}
			path, err := session.SessionFilePath(repo, a.SessionID())
			if err != nil {
				t.Fatal(err)
			}
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			failedRecords := map[string]bool{}
			for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
				var rec struct {
					Type  string `json:"type"`
					Path  string `json:"filePath"`
					Error string `json:"error"`
				}
				if err := json.Unmarshal([]byte(line), &rec); err != nil {
					t.Fatal(err)
				}
				if rec.Type == "review_item_failed" {
					if rec.Error == "" {
						t.Error("failure record has no reason")
					}
					failedRecords[rec.Path] = true
				}
			}
			state, err := session.LoadResumeState(repo, a.SessionID())
			if err != nil {
				t.Fatal(err)
			}
			for _, item := range items {
				if failedRecords[item.Path] != rejected[item.Path] || warningPaths[item.Path] != rejected[item.Path] {
					t.Errorf("%s: failure record=%v warning=%v, rejected=%v",
						item.Path, failedRecords[item.Path], warningPaths[item.Path], rejected[item.Path])
				}
				// A rejected file must not enter the resume index as completed, and a
				// reviewed one must, or the fix would trade a silent gap for a lost checkpoint.
				if _, reusable := state.Item(scanItemFingerprint(item)); reusable == rejected[item.Path] {
					t.Errorf("%s: checkpoint reusable=%v, rejected=%v", item.Path, reusable, rejected[item.Path])
				}
			}
		})
	}
}

func TestExecuteSubtask_RenderedPromptAtLimit(t *testing.T) {
	text := strings.Repeat("word ", 99) + "word"
	count := llm.CountTokens(text)
	// Pick the largest maxTokens whose 80% ceiling still admits this prompt, so the
	// boundary is inclusive: the gate rejects only what exceeds the limit.
	maxTokens := (count*5 + 3) / 4
	if llmloop.PromptTokenLimit(maxTokens) != count {
		t.Fatal("fixture does not match token limit")
	}
	client := &fakeBudgetClient{perCallTokens: 1}
	tpl := budgetTestTemplate()
	tpl.MaxTokens = maxTokens
	tpl.MainTask.Messages = []template.ChatMessage{{Role: "user", Content: text}}
	a := NewAgent(Args{
		Template: tpl, LLMClient: client, Tools: tool.NewRegistry(),
		CommentCollector: tool.NewCommentCollector(), SkipPlan: true,
		Session: session.New(t.TempDir(), "main", "test", session.SessionOptions{ReviewMode: session.ReviewModeFullScan}),
	})
	a.args.Tools.Freeze()
	completed, reason, err := a.executeSubtask(t.Context(), model.ScanItem{Path: "main.go", Content: "package main\n"})
	if err != nil || !completed || reason != "" {
		t.Fatalf("completed=%v reason=%q error=%v", completed, reason, err)
	}
	if got := atomic.LoadInt64(&client.calls); got != 1 {
		t.Errorf("main review calls=%d, want 1", got)
	}
}
