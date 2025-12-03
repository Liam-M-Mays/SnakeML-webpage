#!/usr/bin/env python3
"""
Development server launcher for SnakeML Playground.

This script:
1. Checks for Node.js and npm availability
2. Installs backend dependencies (pip)
3. Installs frontend dependencies (npm)
4. Starts both backend (Flask) and frontend (Vite) servers concurrently

Usage:
    python dev.py [--skip-install] [--backend-only] [--frontend-only]

Options:
    --skip-install     Skip dependency installation (faster restart)
    --backend-only     Only start the backend server
    --frontend-only    Only start the frontend server
"""

import subprocess
import sys
import os
import platform
import shutil
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(msg):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(msg):
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")

def print_error(msg):
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")

def print_info(msg):
    print(f"{Colors.OKCYAN}ℹ {msg}{Colors.ENDC}")

def print_warning(msg):
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")

def check_command(command, name):
    """Check if a command is available in PATH."""
    if shutil.which(command):
        try:
            result = subprocess.run([command, '--version'], capture_output=True, text=True, timeout=5)
            version = result.stdout.split('\n')[0] if result.returncode == 0 else "unknown"
            print_success(f"{name} found: {version}")
            return True
        except Exception as e:
            print_warning(f"{name} found but version check failed: {e}")
            return True
    return False

def check_node_npm():
    """Check for Node.js and npm availability with helpful error messages."""
    print_header("Checking Node.js and npm")

    node_ok = check_command('node', 'Node.js')
    npm_ok = check_command('npm', 'npm')

    if not node_ok or not npm_ok:
        print_error("Node.js and/or npm not found in PATH")
        print_info("\nTo fix this issue:")

        system = platform.system()
        if system == "Darwin":  # macOS
            print_info("  On macOS:")
            print_info("    1. Install via Homebrew: brew install node")
            print_info("    2. Or download from: https://nodejs.org/")
            print_info("    3. If using nvm: run 'nvm use' in this directory")
        elif system == "Windows":
            print_info("  On Windows:")
            print_info("    1. Download from: https://nodejs.org/")
            print_info("    2. Make sure to check 'Add to PATH' during installation")
            print_info("    3. Restart your terminal after installation")
        else:  # Linux
            print_info("  On Linux:")
            print_info("    1. Ubuntu/Debian: sudo apt install nodejs npm")
            print_info("    2. Fedora: sudo dnf install nodejs npm")
            print_info("    3. Or use nvm: https://github.com/nvm-sh/nvm")

        print_info("\n  After installing, restart your terminal and try again.")
        return False

    return True

def install_backend_deps():
    """Install Python dependencies."""
    print_header("Installing Backend Dependencies")

    requirements_file = Path(__file__).parent / "requirements.txt"

    # Check if requirements.txt exists, if not create it
    if not requirements_file.exists():
        print_info("Creating requirements.txt from environment.yml...")
        requirements = [
            "flask>=2.3.0",
            "flask-cors>=4.0.0",
            "flask-socketio>=5.3.0",
            "eventlet>=0.33.0",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "numpy>=1.24.0",
        ]
        requirements_file.write_text("\n".join(requirements))
        print_success("Created requirements.txt")

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            check=True
        )
        print_success("Backend dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install backend dependencies: {e}")
        return False

def install_frontend_deps():
    """Install frontend dependencies via npm."""
    print_header("Installing Frontend Dependencies")

    # Change to repo root where package.json is
    repo_root = Path(__file__).parent

    try:
        # Run npm install from the repo root
        subprocess.run(
            ["npm", "install"],
            cwd=repo_root,
            check=True
        )
        print_success("Frontend dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install frontend dependencies: {e}")
        return False

def start_servers():
    """Start both backend and frontend servers."""
    print_header("Starting Development Servers")

    repo_root = Path(__file__).parent
    backend_dir = repo_root / "backend"

    print_info("Backend will run on: http://127.0.0.1:5000")
    print_info("Frontend will run on: http://localhost:5173")
    print_info("\nPress Ctrl+C to stop both servers\n")

    # Start backend process
    backend_env = os.environ.copy()
    backend_process = subprocess.Popen(
        [sys.executable, "App.py"],
        cwd=backend_dir,
        env=backend_env
    )

    # Start frontend process
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=repo_root
    )

    try:
        # Wait for both processes
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print_info("\n\nShutting down servers...")
        backend_process.terminate()
        frontend_process.terminate()
        backend_process.wait()
        frontend_process.wait()
        print_success("Servers stopped")

def main():
    """Main entry point."""
    skip_install = '--skip-install' in sys.argv
    backend_only = '--backend-only' in sys.argv
    frontend_only = '--frontend-only' in sys.argv

    print_header("🐍 SnakeML Development Server")

    # Always check Node/npm unless backend-only
    if not backend_only:
        if not check_node_npm():
            sys.exit(1)

    # Install dependencies unless skipped
    if not skip_install:
        if not frontend_only:
            if not install_backend_deps():
                print_warning("Continuing despite backend dependency issues...")

        if not backend_only:
            if not install_frontend_deps():
                print_warning("Continuing despite frontend dependency issues...")
    else:
        print_info("Skipping dependency installation (--skip-install)")

    # Start appropriate servers
    if backend_only:
        print_header("Starting Backend Server Only")
        backend_dir = Path(__file__).parent / "backend"
        try:
            subprocess.run([sys.executable, "App.py"], cwd=backend_dir)
        except KeyboardInterrupt:
            print_info("\nBackend server stopped")
    elif frontend_only:
        print_header("Starting Frontend Server Only")
        repo_root = Path(__file__).parent
        try:
            subprocess.run(["npm", "run", "dev"], cwd=repo_root)
        except KeyboardInterrupt:
            print_info("\nFrontend server stopped")
    else:
        start_servers()

if __name__ == "__main__":
    main()
