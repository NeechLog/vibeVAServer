#!/usr/bin/env python3
"""
Wrapper script for vibevaserver-start command.
"""

from audiocloneserver.grpc_server_launcher import run_service_entrypoint


def main() -> None:
    run_service_entrypoint(
        app_main_import="app:main",
        injected_command="start",
        default_log_file="vibevaserver.log",
        default_pid_file="vibevaserver.pid",
    )

if __name__ == "__main__":
    main()
