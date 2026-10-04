// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alibaba/open-code-review/internal/config/template"
	"github.com/alibaba/open-code-review/internal/llm"
	"github.com/alibaba/open-code-review/internal/llmloop"
	"github.com/alibaba/open-code-review/internal/scan"
	"github.com/alibaba/open-code-review/internal/session"
	"github.com/alibaba/open-code-review/internal/tool"
)

type scanPromptLimitClient struct {
	calls int
}

func (c *scanPromptLimitClient) CompletionsWithCtx(_ context.Context, _ llm.ChatRequest) (*llm.ChatResponse, error) {
	c.calls++
	return &llm.ChatResponse{
		Choices: []llm.Choice{{
			Message: llm.ResponseMessage{
				Role: "assistant",
				ToolCalls: []llm.ToolCall{{
					ID:       "done",
					Type:     "function",
					Function: llm.FunctionCall{Name: "task_done", Arguments: "{}"},
				}},
			},
			FinishReason: "tool_calls",
		}},
	}, nil
}

func TestScanRenderedPromptLimitOutput(t *testing.T) {
	for _, tc := range []struct {
		name      string
		mixed     bool
		wantCalls int
	}{
		{name: "all files rejected", wantCalls: 0},
		{name: "one file completes", mixed: true, wantCalls: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			setTestHome(t, t.TempDir())
			repoDir := t.TempDir()
			contents := map[string]string{
				"rejected.go": "package x\n// " + strings.Repeat("value ", 30) + "\n",
			}
			if tc.mixed {
				contents["completed.go"] = "package x\n"
			}

			const maxTokens = 100
			limit := llmloop.PromptTokenLimit(maxTokens)
			for name, content := range contents {
				// Every file must pass selection so only the rendered prompt gate can reject it.
				if tokens := llm.CountTokens(content); tokens > limit {
					t.Fatalf("%s content tokens = %d, exceed selection limit %d", name, tokens, limit)
				}
				rendered := []llm.Message{llm.NewTextMessage("user", strings.Repeat(content, 4))}
				tooLarge := llmloop.CountMessagesTokens(rendered) > limit
				if tooLarge != (name == "rejected.go") {
					t.Fatalf("%s rendered prompt rejection = %v", name, tooLarge)
				}
				if err := os.WriteFile(filepath.Join(repoDir, name), []byte(content), 0o600); err != nil {
					t.Fatalf("write %s: %v", name, err)
				}
			}

			client := &scanPromptLimitClient{}
			ag := scan.NewAgent(scan.Args{
				RepoDir: repoDir,
				Template: template.ScanTemplate{
					MaxTokens:           maxTokens,
					MaxToolRequestTimes: 5,
					MainTask: template.LlmConversation{
						Messages: []template.ChatMessage{{Role: "user", Content: strings.Repeat("{{file_content}}", 4)}},
					},
				},
				LLMClient:        client,
				Tools:            tool.NewRegistry(),
				CommentCollector: tool.NewCommentCollector(),
				MaxConcurrency:   1,
				Session:          session.New(repoDir, "main", "test", session.SessionOptions{ReviewMode: session.ReviewModeFullScan}),
				SkipPlan:         true,
				SkipDedup:        true,
				SkipSummary:      true,
			})

			comments, err := ag.Run(context.Background())
			if tc.mixed {
				if err != nil {
					t.Fatalf("mixed scan must succeed: %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), "all 1 file scan(s) failed") {
				t.Fatalf("all-rejected scan error = %v, want all-files failure", err)
			}
			if client.calls != tc.wantCalls {
				t.Fatalf("MAIN_TASK calls = %d, want %d", client.calls, tc.wantCalls)
			}
			if len(comments) != 0 {
				t.Fatalf("comments = %v, want none", comments)
			}

			var thresholdWarning, failureWarning bool
			for _, warning := range ag.Warnings() {
				if warning.File != "rejected.go" {
					t.Errorf("unexpected warning for %q: %+v", warning.File, warning)
				}
				switch warning.Type {
				case "token_threshold_exceeded":
					thresholdWarning = true
				case "scan_subtask_error":
					failureWarning = true
					if !strings.Contains(warning.Message, "rendered prompt") {
						t.Errorf("failure warning = %q, want rendered prompt reason", warning.Message)
					}
				}
			}
			if !thresholdWarning || !failureWarning {
				t.Fatalf("warnings = %+v, want threshold and subtask failure", ag.Warnings())
			}

			raw := captureStdout(t, func() {
				if err := emitRunResult(context.Background(), ag, comments, time.Now(), "json", "developer", nil, nil, os.Stdout, nil); err != nil {
					t.Fatalf("emit JSON: %v", err)
				}
			})
			var out jsonOutput
			if err := json.Unmarshal([]byte(raw), &out); err != nil {
				t.Fatalf("unmarshal JSON: %v\n%s", err, raw)
			}
			if out.Status != "completed_with_errors" {
				t.Errorf("status = %q, want completed_with_errors", out.Status)
			}
			if !strings.Contains(out.Message, "Some files could not be reviewed") || strings.Contains(raw, "Looks good to me") {
				t.Errorf("JSON output must report incomplete review: %s", raw)
			}

			text := captureStdout(t, func() {
				if err := emitRunResult(context.Background(), ag, comments, time.Now(), "text", "developer", nil, nil, os.Stdout, nil); err != nil {
					t.Fatalf("emit text: %v", err)
				}
			})
			if !strings.Contains(text, "Some files could not be reviewed") || strings.Contains(text, "Looks good to me") {
				t.Errorf("text output must report incomplete review: %q", text)
			}
		})
	}
}

// TestScanRenderedPromptLimitExitCode drives the real CLI entry point in a child
// process. A scan whose only file is rejected by the rendered-prompt gate must
// exit non-zero, must say that the file scan failed, and must never print a clean
// review — the failure path returns before any output payload is emitted.
func TestScanRenderedPromptLimitExitCode(t *testing.T) {
	if repo := os.Getenv("OCR_TEST_SCAN_PROMPT_LIMIT_REPO"); repo != "" {
		rootCmd.SetArgs([]string{
			"scan", "--repo", repo,
			"--max-tokens", "100",
			"--no-plan", "--no-dedup", "--no-summary",
			"--format", "json",
		})
		os.Exit(run())
	}

	var requests atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		http.Error(w, "a rejected file must never reach the provider", http.StatusInternalServerError)
	}))
	defer server.Close()

	repo := t.TempDir()
	retryTestGit(t, repo, "init", "-q")
	if err := os.WriteFile(filepath.Join(repo, "main.go"), []byte("package main\n"), 0o600); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	retryTestGit(t, repo, "add", ".")
	retryTestGit(t, repo, "commit", "-q", "-m", "base")

	home := t.TempDir()
	child := exec.Command(os.Args[0], "-test.run=^TestScanRenderedPromptLimitExitCode$")
	child.Env = append(os.Environ(),
		"OCR_TEST_SCAN_PROMPT_LIMIT_REPO="+repo,
		"HOME="+home,
		"USERPROFILE="+home,
		"XDG_CONFIG_HOME="+filepath.Join(home, ".config"),
		"OCR_LLM_URL="+server.URL,
		"OCR_LLM_TOKEN=test-token",
		"OCR_LLM_MODEL=test-model",
		"OCR_LLM_PROTOCOL=openai",
	)
	output, err := child.CombinedOutput()

	exitErr, ok := err.(*exec.ExitError)
	if !ok || exitErr.ExitCode() != 1 {
		t.Fatalf("exit error = %v, want exit code 1; output:\n%s", err, output)
	}
	if !strings.Contains(string(output), "file scan(s) failed") {
		t.Errorf("output must name the all-files failure; got:\n%s", output)
	}
	if strings.Contains(string(output), "Looks good to me") {
		t.Errorf("a scan that reviewed nothing must not report a clean result; got:\n%s", output)
	}
	if got := requests.Load(); got != 0 {
		t.Errorf("provider requests = %d, want 0", got)
	}
}
