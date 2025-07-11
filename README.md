# SDRFscribe Project Setup and Usage Guide

This guide will walk you through setting up the SDRFscribe project, configuring the necessary servers, and finally generating SDRF files.

## Prerequisites

Before you begin, ensure your system meets the following requirements:

*   **Python 3.8+** (uv requires a Python runtime)
*   **VS Code** (Visual Studio Code)
*   **Git** (Optional, if cloning the project from a Git repository)
*   **Windows Operating System** (Commands in this guide are primarily for Windows PowerShell)

## Step 1: Install uv and Set Up Your Project Environment

`uv` is a high-performance Python package manager and bundler.

1.  **Install uv:**
    Run the following command in PowerShell to install `uv`:

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    *   **Note**: This command is specific to Windows PowerShell. For macOS or Linux, please refer to the `uv` official documentation for installation.

2.  **Create and Navigate to Your Project Directory:**
    ```bash
    mkdir SDRFscribe
    cd SDRFscribe
    ```

3.  **Initialize uv Project:**
    Inside the `SDRFscribe` directory, run `uv init` to initialize the project structure.
    ```bash
    uv init
    ```

4.  **Create and Activate Virtual Environment:**
    This step creates an isolated Python environment to prevent dependency conflicts.
    ```bash
    uv venv
    .venv\Scripts\activate
    ```
    *   **Note**: `.venv\Scripts\activate` is the activation command for Windows. On macOS/Linux, you should use `source .venv/bin/activate`.
    *   Upon successful activation, your command prompt will show `(.venv)` prefix, indicating you are in the virtual environment.

## Step 2: Install Python Dependencies

With the virtual environment activated, use the `uv add` command to install all necessary project dependencies.

```bash
uv add mcp[cli] httpx beautifulsoup4 google-generativeai lxml
```

*   These packages provide core command-line interfaces, HTTP request capabilities, HTML parsing functionalities, Google Gemini API interaction, and XML/HTML processing capabilities.

## Step 3: Configure Your Google API Key

The `pdf_processor_server.py` script will use this API key to interact with the Google Gemini API. Setting the API key as an environment variable is a best practice for securing sensitive information.

1.  **Set the `GOOGLE_API_KEY` Environment Variable:**
    Run the following command in PowerShell to set your Google API key.

    ```powershell
    setx GOOGLE_API_KEY="YOUR_ACTUAL_API_KEY"
    ```
    *   **Important**: Replace `"YOUR_ACTUAL_API_KEY"` with your actual Google API key.
    *   The `setx` command permanently sets the environment variable for your user.
    *   **After setting, you must close your current command prompt/PowerShell window and open a new one for the environment variable to take effect.**
    *   **Security Warning**: Never hardcode your API key directly into your source code, and never commit it to any version control system (e.g., Git).

## Step 4: Configure VS Code MCP Servers

This section guides you through configuring the VS Code extension to run the required MCP (Message Communication Protocol) servers.

1.  **Install VS Code Extension:**
    Open VS Code, go to the Extensions view (`Ctrl+Shift+X`), search for and install the extension named `cline`.

2.  **Open MCP Server Configuration:**
    In VS Code, you typically open settings via:
    *   Go to **File** -> **Preferences** -> **Settings**.
    *   Search for "MCP Servers" or look for the `settings.json` file.
    *   Alternatively, you might find a specific configuration entry for the `cline` extension.

3.  **Paste Server Configuration:**
    Add or replace the existing configuration in your MCP server settings with the following JSON structure.

    ```json
    {
      "mcpServers": {
        "pride_explorer": {
          "disabled": false,
          "timeout": 600,
          "type": "stdio",
          "command": "uv",
          "args": [
            "run",
            "pathto/SDRFscribe/.venv/Scripts/python.exe",
            "pathto/SDRFscribe/pride_server.py"
          ]
        },
        "ontology_search": {
          "disabled": false,
          "timeout": 600,
          "type": "stdio",
          "command": "uv",
          "args": [
            "run",
            "pathto/SDRFscribe/.venv/Scripts/python.exe",
            "pathto/SDRFscribe/ontology_server.py"
          ]
        },
        "pdf_processor": {
          "disabled": false,
          "timeout": 6000,
          "type": "stdio",
          "command": "uv",
          "args": [
            "run",
            "pathto/SDRFscribe/.venv/Scripts/python.exe",
            "pathto/SDRFscribe/pdf_processor_server.py"
          ]
        }
      }
    }
    ```
    *   **Crucial**:
        *   Replace all instances of `pathto/SDRFscribe` with the **absolute path** to your `SDRFscribe` project directory. For example, if your project is in `C:\Users\YourUser\Documents\SDRFscribe`, the path should be `C:/Users/YourUser/Documents/SDRFscribe` (note the forward slashes in JSON).
        *   `pride_explorer`, `ontology_search`, and `pdf_processor` are three distinct server configurations, each responsible for different backend functionalities.
        *   The `timeout` for `pdf_processor` is set to `6000` seconds (100 minutes), likely because it may need to process larger PDF files.

## Step 5: Verify Server Status

After configuring the MCP servers in VS Code, you need to ensure they have started correctly and are running.

*   **Indicator of Success**: The servers are successfully started and ready when the **MCPServer displays a green light**.
    *   You can observe the server status indicator within the `cline` extension's interface or its associated output panel.

## Step 6: Generate the SDRF File

Once all MCP servers are showing a green light, you can proceed to generate your SDRF file.

1.  **Open the `cline` extension interface.**
2.  **Prepare Input Files:** Drag and drop the `README.adoc`, `template.csv`, and `prompt.txt` files into `cline`'s chat interface or designated input area.
3.  **Start Generation:** After submitting these files, the process of generating the SDRF file will begin.
