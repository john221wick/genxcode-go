package cmd

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/john221wick/genxcode-go/pkg/config"
	"github.com/john221wick/genxcode-go/pkg/generator"
	"github.com/john221wick/genxcode-go/pkg/template"
	"github.com/spf13/cobra"
)

var (
	forceFlag bool
	repoOwner string
	repoName  string
	repoBranch string
)

func newManager() *template.Manager {
	if repoOwner != "" && repoName != "" {
		branch := repoBranch
		if branch == "" {
			branch = "main"
		}
		return template.NewManagerWithRepo(repoOwner, repoName, branch)
	}
	return template.NewManager()
}

var rootCmd = &cobra.Command{
	Use:     "genxcode",
	Short:   "Generate boilerplate code from remote templates",
	Long:    `genxcode fetches templates from GitHub, caches them locally, and renders projects from YAML configs.`,
	Version: config.Version,
}

var initCmd = &cobra.Command{
	Use:   "init <template>",
	Short: "Create a genxcode.yaml for a template",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		mgr := newManager()
		templateName := strings.TrimSpace(args[0])

		if err := mgr.EnsureAvailable(templateName); err != nil {
			return fmt.Errorf("failed to fetch template: %w", err)
		}

		tmplDir := mgr.TemplateDir(templateName)
		spec, err := config.LoadTemplateSpec(filepath.Join(tmplDir, "template.yaml"))
		if err != nil {
			return err
		}

		defaultConfigPath := filepath.Join(tmplDir, spec.DefaultConfigFilename)
		dest := config.DefaultConfigFilename
		if err := config.WriteDefaultConfig(dest, templateName, defaultConfigPath, forceFlag); err != nil {
			return err
		}
		fmt.Printf("Created %s\n", dest)
		if spec.ApplyHint != "" {
			fmt.Printf("  %s\n", spec.ApplyHint)
		}
		return nil
	},
}

var applyCmd = &cobra.Command{
	Use:   "apply",
	Short: "Generate boilerplate from genxcode.yaml",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg, err := config.LoadConfig(config.DefaultConfigFilename)
		if err != nil {
			return err
		}

		mgr := newManager()
		if err := mgr.EnsureAvailable(cfg.Template); err != nil {
			return fmt.Errorf("failed to fetch template: %w", err)
		}

		tmplDir := mgr.TemplateDir(cfg.Template)
		spec, err := config.LoadTemplateSpec(filepath.Join(tmplDir, "template.yaml"))
		if err != nil {
			return err
		}

		// Merge defaults
		defaultPath := filepath.Join(tmplDir, spec.DefaultConfigFilename)
		defaultData, err := loadYAMLMap(defaultPath)
		if err != nil {
			return fmt.Errorf("load defaults: %w", err)
		}
		merged := mergeMaps(defaultData, cfg.Data)
		merged["template"] = cfg.Template
		merged["name"] = cfg.Name

		ctx, err := generator.BuildContext(merged)
		if err != nil {
			return fmt.Errorf("build context: %w", err)
		}

		outputDir := filepath.Join(".", cfg.Name)
		fmt.Printf("Generating %s (%s) in %s\n", cfg.Name, cfg.Template, outputDir)
		if err := generator.GenerateProject(tmplDir, outputDir, ctx); err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		fmt.Println()
		if spec.ApplyHint != "" {
			fmt.Printf("  %s\n", spec.ApplyHint)
		}
		return nil
	},
}

var updateCmd = &cobra.Command{
	Use:   "update",
	Short: "Update genxcode to the latest release",
	RunE: func(cmd *cobra.Command, args []string) error {
		return selfUpdate()
	},
}

func selfUpdate() error {
	currentVer := strings.TrimPrefix(config.Version, "v")
	if currentVer == "" || currentVer == "dev" {
		fmt.Println("You are running a development build. Skipping self-update.")
		return nil
	}

	// Fetch latest release tag
	apiURL := "https://api.github.com/repos/john221wick/genxcode-go/releases/latest"
	resp, err := http.Get(apiURL)
	if err != nil {
		return fmt.Errorf("failed to check for updates: %w", err)
	}
	defer resp.Body.Close()

	var release struct {
		TagName string `json:"tag_name"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&release); err != nil {
		return fmt.Errorf("failed to parse release info: %w", err)
	}

	latestVer := strings.TrimPrefix(release.TagName, "v")
	if latestVer == currentVer {
		fmt.Printf("genxcode is already up to date (%s).\n", currentVer)
		return nil
	}

	fmt.Printf("Current: v%s  Latest: v%s\n", currentVer, latestVer)

	// Detect OS/arch
	goos := runtime.GOOS
	goarch := runtime.GOARCH
	if goarch == "amd64" {
		goarch = "amd64"
	}

	ext := "tar.gz"
	if goos == "windows" {
		ext = "zip"
	}

	assetName := fmt.Sprintf("genxcode_%s_%s_%s.%s", latestVer, goos, goarch, ext)
	downloadURL := fmt.Sprintf("https://github.com/john221wick/genxcode-go/releases/download/v%s/%s", latestVer, assetName)

	fmt.Printf("Downloading %s...\n", assetName)

	// Download to temp file
	tmpFile, err := os.CreateTemp("", "genxcode-update-*")
	if err != nil {
		return fmt.Errorf("create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name())

	dlResp, err := http.Get(downloadURL)
	if err != nil {
		return fmt.Errorf("download failed: %w", err)
	}
	defer dlResp.Body.Close()
	if dlResp.StatusCode != 200 {
		return fmt.Errorf("download failed: HTTP %s (asset may not exist for your platform)", dlResp.Status)
	}

	if _, err := io.Copy(tmpFile, dlResp.Body); err != nil {
		return fmt.Errorf("write download: %w", err)
	}
	tmpFile.Close()

	// Extract if archive
	extractedPath := tmpFile.Name()
	if ext == "tar.gz" {
		extractDir, err := os.MkdirTemp("", "genxcode-extract-*")
		if err != nil {
			return fmt.Errorf("create extract dir: %w", err)
		}
		defer os.RemoveAll(extractDir)

		cmd := exec.Command("tar", "-xzf", tmpFile.Name(), "-C", extractDir)
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("extract archive: %w", err)
		}
		extractedPath = filepath.Join(extractDir, "genxcode")
	}

	if err := os.Chmod(extractedPath, 0755); err != nil {
		return fmt.Errorf("chmod: %w", err)
	}

	// Find current binary path
	currentPath, err := os.Executable()
	if err != nil {
		return fmt.Errorf("find current binary: %w", err)
	}
	currentPath, err = filepath.EvalSymlinks(currentPath)
	if err != nil {
		return fmt.Errorf("resolve binary path: %w", err)
	}

	// Replace binary: rename old, move new
	oldPath := currentPath + ".old"
	if err := os.Rename(currentPath, oldPath); err != nil {
		return fmt.Errorf("backup old binary: %w", err)
	}
	if err := os.Rename(extractedPath, currentPath); err != nil {
		// Try to restore old
		os.Rename(oldPath, currentPath)
		return fmt.Errorf("install new binary: %w", err)
	}
	os.Remove(oldPath)

	fmt.Printf("Updated genxcode to v%s\n", latestVer)
	return nil
}

var listCmd = &cobra.Command{
	Use:   "list",
	Short: "List cached templates",
	RunE: func(cmd *cobra.Command, args []string) error {
		mgr := newManager()
		cached, err := mgr.ListCached()
		if err != nil {
			return err
		}
		if len(cached) == 0 {
			fmt.Println("No cached templates. Run `genxcode init <template>` to download one.")
			return nil
		}
		fmt.Println("Available templates:")
		for _, name := range cached {
			tmplDir := mgr.TemplateDir(name)
			spec, err := config.LoadTemplateSpec(filepath.Join(tmplDir, "template.yaml"))
			if err != nil {
				fmt.Printf("  - %s (error reading spec: %v)\n", name, err)
				continue
			}
			desc := ""
			if spec.Description != "" {
				desc = fmt.Sprintf(": %s", spec.Description)
			}
			aliases := ""
			if len(spec.Aliases) > 0 {
				aliases = fmt.Sprintf(" (aliases: %s)", strings.Join(spec.Aliases, ", "))
			}
			fmt.Printf("  - %s%s%s\n", spec.Name, aliases, desc)
		}
		return nil
	},
}

var removeCmd = &cobra.Command{
	Use:   "remove",
	Short: "Remove genxcode.yaml from current directory",
	RunE: func(cmd *cobra.Command, args []string) error {
		path := config.DefaultConfigFilename
		if _, err := os.Stat(path); os.IsNotExist(err) {
			return fmt.Errorf("%s does not exist", path)
		}
		if err := os.Remove(path); err != nil {
			return fmt.Errorf("failed to remove %s: %w", path, err)
		}
		fmt.Printf("Removed %s\n", path)
		return nil
	},
}

func init() {
	rootCmd.PersistentFlags().StringVar(&repoOwner, "owner", "", "GitHub owner for remote templates")
	rootCmd.PersistentFlags().StringVar(&repoName, "repo", "", "GitHub repo for remote templates")
	rootCmd.PersistentFlags().StringVar(&repoBranch, "branch", "", "GitHub branch for remote templates")

	initCmd.Flags().BoolVar(&forceFlag, "force", false, "Overwrite existing config")

	rootCmd.AddCommand(initCmd)
	rootCmd.AddCommand(applyCmd)
	rootCmd.AddCommand(updateCmd)
	rootCmd.AddCommand(listCmd)
	rootCmd.AddCommand(removeCmd)
}

// Execute runs the root command.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func loadYAMLMap(path string) (map[string]interface{}, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m map[string]interface{}
	if err := config.ParseYAML(data, &m); err != nil {
		return nil, err
	}
	return m, nil
}

func mergeMaps(base, overlay map[string]interface{}) map[string]interface{} {
	out := make(map[string]interface{})
	for k, v := range base {
		out[k] = v
	}
	for k, v := range overlay {
		out[k] = v
	}
	return out
}
