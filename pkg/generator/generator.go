package generator

import (
	"bytes"
	"encoding/json"
	"fmt"
	"text/template"
	"os"
	"path/filepath"
	"strings"
)

// Context holds all data passed to templates.
type Context struct {
	Name      string
	Template  string
	Arch      string
	InputDim  int
	OutputDim int
	Hidden    []int
	Activation string
	Metrics   []string
	Prod      bool
	LR        float64
	Epochs    int
	BatchSize int
	Device    string
	Data      map[string]interface{} // raw data for any extra fields

	// Computed fields
	HasF1        bool
	HasPrecision bool
	HasRecall    bool
	HasAccuracy  bool
	HasLoss       bool
	NeedsSklearn  bool
	SklearnImports string
	Layers        []Layer
	OutputLayerIn int
}

// Layer represents a single MLP layer for models.py.
type Layer struct {
	In  int
	Out int
}

// BuildContext creates a fully populated template context from raw config data.
func BuildContext(data map[string]interface{}) (*Context, error) {
	ctx := &Context{
		Data: data,
	}

	// Extract fields with defaults
	ctx.Name = getString(data, "name", "project")
	ctx.Template = getString(data, "template", "pytorch")
	ctx.Arch = getString(data, "arch", "mlp")
	ctx.InputDim = getInt(data, "input_dim", 784)
	ctx.OutputDim = getInt(data, "output_dim", 10)
	ctx.Hidden = getIntSlice(data, "hidden", []int{256, 128})
	ctx.Activation = getString(data, "activation", "relu")
	ctx.Metrics = getStringSlice(data, "metrics", []string{"accuracy", "loss"})
	ctx.Prod = getBool(data, "prod", false)
	ctx.LR = getFloat(data, "lr", 0.001)
	ctx.Epochs = getInt(data, "epochs", 10)
	ctx.BatchSize = getInt(data, "batch_size", 32)
	ctx.Device = getString(data, "device", "cpu")

	// Compute booleans
	for _, m := range ctx.Metrics {
		switch m {
		case "f1":
			ctx.HasF1 = true
		case "precision":
			ctx.HasPrecision = true
		case "recall":
			ctx.HasRecall = true
		case "accuracy":
			ctx.HasAccuracy = true
		case "loss":
			ctx.HasLoss = true
		}
	}
	ctx.NeedsSklearn = ctx.HasF1 || ctx.HasPrecision || ctx.HasRecall

	// Build sklearn import line
	var sklearnFuncs []string
	if ctx.HasF1 {
		sklearnFuncs = append(sklearnFuncs, "f1_score")
	}
	if ctx.HasPrecision {
		sklearnFuncs = append(sklearnFuncs, "precision_score")
	}
	if ctx.HasRecall {
		sklearnFuncs = append(sklearnFuncs, "recall_score")
	}
	ctx.SklearnImports = strings.Join(sklearnFuncs, ", ")

	// Build layers for models.py
	prev := ctx.InputDim
	for _, h := range ctx.Hidden {
		ctx.Layers = append(ctx.Layers, Layer{In: prev, Out: h})
		prev = h
	}
	ctx.OutputLayerIn = prev

	return ctx, nil
}

// TemplateFuncs returns custom functions available in templates.
func TemplateFuncs() template.FuncMap {
	return template.FuncMap{
		"json": func(v interface{}) string {
			b, _ := json.Marshal(v)
			return string(b)
		},
		"contains": contains,
		"join": func(sep string, vals []string) string {
			return strings.Join(vals, sep)
		},
		"last": func(i, len int) bool {
			return i == len-1
		},
		"upper": strings.ToUpper,
		"split": func(s string, sep string) []string {
			return strings.Split(s, sep)
		},
	}
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// Render renders a single template string with the context.
func Render(tmplStr string, ctx *Context) (string, error) {
	tmpl, err := template.New("fragment").Funcs(TemplateFuncs()).Parse(tmplStr)
	if err != nil {
		return "", fmt.Errorf("parse template: %w", err)
	}
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, ctx); err != nil {
		return "", fmt.Errorf("execute template: %w", err)
	}
	return buf.String(), nil
}

// GenerateProject walks template files and renders them to output directory.
func GenerateProject(templateDir, outputDir string, ctx *Context) error {
	filesDir := filepath.Join(templateDir, "files")
	return filepath.Walk(filesDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}

		rel, err := filepath.Rel(filesDir, path)
		if err != nil {
			return err
		}

		// Render the relative path itself (for filenames like {{.Name}}Dataset.py)
		isTemplate := strings.HasSuffix(rel, ".j2") || strings.HasSuffix(rel, ".tmpl")
		targetRel := rel
		if isTemplate {
			targetRel = targetRel[:len(targetRel)-3] // strip .j2 or .tmpl
		}

		renderedRel, err := Render(targetRel, ctx)
		if err != nil {
			return fmt.Errorf("render filename: %w", err)
		}

		targetPath := filepath.Join(outputDir, renderedRel)
		if err := os.MkdirAll(filepath.Dir(targetPath), 0755); err != nil {
			return err
		}

		if isTemplate {
			content, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			rendered, err := Render(string(content), ctx)
			if err != nil {
				return fmt.Errorf("render %s: %w", rel, err)
			}
			if err := os.WriteFile(targetPath, []byte(rendered), 0644); err != nil {
				return err
			}
		} else {
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			if err := os.WriteFile(targetPath, data, 0644); err != nil {
				return err
			}
		}

		fmt.Printf("  %s\n", renderedRel)
		return nil
	})
}

// Helper getters

func getString(m map[string]interface{}, key, def string) string {
	v, ok := m[key]
	if !ok {
		return def
	}
	s, ok := v.(string)
	if ok {
		return s
	}
	return fmt.Sprintf("%v", v)
}

func getInt(m map[string]interface{}, key string, def int) int {
	v, ok := m[key]
	if !ok {
		return def
	}
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	}
	return def
}

func getFloat(m map[string]interface{}, key string, def float64) float64 {
	v, ok := m[key]
	if !ok {
		return def
	}
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	case int64:
		return float64(n)
	}
	return def
}

func getBool(m map[string]interface{}, key string, def bool) bool {
	v, ok := m[key]
	if !ok {
		return def
	}
	b, ok := v.(bool)
	if ok {
		return b
	}
	return def
}

func getIntSlice(m map[string]interface{}, key string, def []int) []int {
	v, ok := m[key]
	if !ok {
		return def
	}
	slice, ok := v.([]interface{})
	if !ok {
		return def
	}
	var out []int
	for _, item := range slice {
		switch n := item.(type) {
		case int:
			out = append(out, n)
		case int64:
			out = append(out, int(n))
		case float64:
			out = append(out, int(n))
		}
	}
	return out
}

func getStringSlice(m map[string]interface{}, key string, def []string) []string {
	v, ok := m[key]
	if !ok {
		return def
	}
	slice, ok := v.([]interface{})
	if !ok {
		return def
	}
	var out []string
	for _, item := range slice {
		s, ok := item.(string)
		if ok {
			out = append(out, s)
		}
	}
	return out
}
