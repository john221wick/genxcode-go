package cmd

import (
	"fmt"
	"os"
	"path/filepath"
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
	Use:   "update [template]",
	Short: "Update cached templates from remote",
	Args:  cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		mgr := newManager()
		if len(args) > 0 {
			name := args[0]
			fmt.Printf("Updating template %s...\n", name)
			if err := mgr.Fetch(name); err != nil {
				return err
			}
			fmt.Printf("Updated %s\n", name)
			return nil
		}

		// Update all cached
		cached, err := mgr.ListCached()
		if err != nil {
			return err
		}
		if len(cached) == 0 {
			fmt.Println("No cached templates. Run `genxcode init <template>` first.")
			return nil
		}
		for _, name := range cached {
			fmt.Printf("Updating template %s...\n", name)
			if err := mgr.Fetch(name); err != nil {
				fmt.Fprintf(os.Stderr, "  failed: %v\n", err)
				continue
			}
			fmt.Printf("  done\n")
		}
		return nil
	},
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

func init() {
	rootCmd.PersistentFlags().StringVar(&repoOwner, "owner", "", "GitHub owner for remote templates")
	rootCmd.PersistentFlags().StringVar(&repoName, "repo", "", "GitHub repo for remote templates")
	rootCmd.PersistentFlags().StringVar(&repoBranch, "branch", "", "GitHub branch for remote templates")

	initCmd.Flags().BoolVar(&forceFlag, "force", false, "Overwrite existing config")

	rootCmd.AddCommand(initCmd)
	rootCmd.AddCommand(applyCmd)
	rootCmd.AddCommand(updateCmd)
	rootCmd.AddCommand(listCmd)
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
