# genxcode

Generate boilerplate code from templates.

## Install

```bash
curl -sSfL https://raw.githubusercontent.com/john221wick/genxcode-go/main/install.sh | sh
```

Or with Go:

```bash
go install github.com/john221wick/genxcode-go@latest
```

## Usage

```bash
# Create a config file for a template
genxcode init pytorch

# Edit genxcode.yaml to your needs, then generate the project
genxcode apply

# List cached templates
genxcode list

# Update templates from remote
genxcode update

# Update a specific template
genxcode update pytorch
```

Templates are downloaded from GitHub on first use and cached in `~/.genxcode/templates/`.
