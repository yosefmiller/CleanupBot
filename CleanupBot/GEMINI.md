# Project Guidelines

## Pixi Environment (WSL)
- **Binary Location**: `/home/yosef/.pixi/bin/pixi` (requires host filesystem access outside the workspace sandbox).
- For any component containing `pixi.toml` (such as `VisionMCP/` or root `src/`), always navigate to that directory and invoke Pixi using:
  ```bash
  cd <dir_with_pixi_toml> && PATH="/home/yosef/.pixi/bin:$PATH" /home/yosef/.pixi/bin/pixi run <command>
  ```
- **Common commands**:
  - Run tasks: `cd VisionMCP && PATH="/home/yosef/.pixi/bin:$PATH" /home/yosef/.pixi/bin/pixi run <task>` (e.g. `check-cuda`, `serve`)
  - Run python: `cd VisionMCP && PATH="/home/yosef/.pixi/bin:$PATH" /home/yosef/.pixi/bin/pixi run python <script.py>`
  - Inspect environment: `cd VisionMCP && PATH="/home/yosef/.pixi/bin:$PATH" /home/yosef/.pixi/bin/pixi info`
  - List packages: `cd VisionMCP && PATH="/home/yosef/.pixi/bin:$PATH" /home/yosef/.pixi/bin/pixi list`
