package google_test

import (
	"context"
	"net/http"
	"os"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/google"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

const recordedGoogleProject = "recorded-project"

func recordedGoogleModel(t *testing.T, cassetteName, modelName string) *google.Model {
	t.Helper()
	mode := recorder.ModeReplayOnly
	project := recordedGoogleProject
	var tokenProvider google.TokenProvider = func(context.Context) (string, error) {
		return "recorded-token", nil
	}
	if _, err := os.Stat("testdata/" + cassetteName + ".yaml"); os.IsNotExist(err) {
		project = os.Getenv("GOOGLE_CLOUD_PROJECT")
		if project == "" {
			t.Skip("set GOOGLE_CLOUD_PROJECT and Application Default Credentials to record this cassette")
		}
		mode = recorder.ModeRecordOnce
		tokenProvider = nil
	}
	recording, err := recorder.New("testdata/"+cassetteName,
		recorder.WithMode(mode),
		recorder.WithHook(func(interaction *cassette.Interaction) error {
			delete(interaction.Request.Headers, "Authorization")
			delete(interaction.Request.Headers, "X-Goog-Api-Key")
			delete(interaction.Response.Headers, "Set-Cookie")
			interaction.Request.URL = strings.ReplaceAll(interaction.Request.URL, project, recordedGoogleProject)
			return nil
		}, recorder.AfterCaptureHook),
		recorder.WithMatcher(cassette.MatcherFunc(func(request *http.Request, interaction cassette.Request) bool {
			return request.Method == interaction.Method && request.URL.String() == interaction.URL
		})),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := recording.Stop(); err != nil {
			t.Error(err)
		}
	})
	model, err := google.NewVertexModel(modelName, google.VertexConfig{
		Project: project, HTTPClient: recording.GetDefaultClient(), TokenProvider: tokenProvider,
	})
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func recordedGooglePrompt(content string) []ai.ModelMessage {
	return []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: content}}}}
}

func TestRecordedGemini25Search(t *testing.T) {
	response, err := ai.RequestModel(t.Context(), recordedGoogleModel(t, "gemini_25_search", "gemini-2.5-flash"),
		recordedGooglePrompt("Search the web for the current Go release and cite a source."),
		ai.ModelRequestParams{AllowText: true, NativeTools: []ai.NativeTool{ai.WebSearchTool{}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if response.Text() == "" || response.ProviderDetails["grounding_metadata"] == nil {
		t.Fatalf("expected grounded Gemini 2.5 response, got %#v", response)
	}
}

func TestRecordedGemini38Search(t *testing.T) {
	response, err := ai.RequestModel(t.Context(), recordedGoogleModel(t, "gemini_38_search", "gemini-3.8-flash"),
		recordedGooglePrompt("Search the web for the current Go release and cite a source."),
		ai.ModelRequestParams{AllowText: true, NativeTools: []ai.NativeTool{ai.WebSearchTool{}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if response.Text() == "" || response.ProviderDetails["grounding_metadata"] == nil {
		t.Fatalf("expected grounded Gemini 3.8 response, got %#v", response)
	}
}

func TestRecordedGemini38StreamingSearch(t *testing.T) {
	stream := ai.StreamModel(t.Context(), recordedGoogleModel(t, "gemini_38_streaming_search", "gemini-3.8-flash"),
		recordedGooglePrompt("Search the web for the current Go release and cite a source."),
		ai.ModelRequestParams{AllowText: true, NativeTools: []ai.NativeTool{ai.WebSearchTool{}}},
	)
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	response := stream.Response()
	if response == nil || response.Text() == "" || response.ProviderDetails["grounding_metadata"] == nil {
		t.Fatalf("expected grounded streamed Gemini 3.8 response, got %#v", response)
	}
}

func TestRecordedGemini38SearchWithFunction(t *testing.T) {
	response, err := ai.RequestModel(t.Context(), recordedGoogleModel(t, "gemini_38_search_with_function", "gemini-3.8-flash"),
		recordedGooglePrompt("Search the web for the current Go release, then use record_result for one source."),
		ai.ModelRequestParams{
			AllowText:   true,
			NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
			Tools: []ai.ToolDefinition{{
				Name: "record_result", Description: "Record a web search result.",
				Schema: map[string]any{
					"type": "object", "properties": map[string]any{"title": map[string]any{"type": "string"}},
				},
			}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if response.Text() == "" && len(response.ToolCalls()) == 0 {
		t.Fatalf("expected a Gemini 3.8 response for combined tools, got %#v", response)
	}
}
