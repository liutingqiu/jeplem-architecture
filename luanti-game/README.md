# PLWM Luanti Training World

This directory contains a self-owned Luanti game and its `plwm_training` Lua
mod. It does not reuse Minecraft assets, code, accounts, or network services.

## License boundary

The project uses the official Luanti 5.16.1 Windows release. Luanti source code
is LGPL-2.1-or-later; the release also includes its own texture and sound
license notices. This game contains only PLWM-authored Lua and uses Luanti's
bundled `blank.png` with color modifiers at runtime.

## Capability boundary

Python and the Lua mod communicate only through JSON Lines under the configured
`plwm_bridge_path`. The mod accepts `observe`, `set_episode`, `reset_episode`,
and cardinal `step` actions. It never accepts shell commands, Lua source,
network requests, arbitrary console commands, block placement, or file paths.

Each action has a validated `request_id`. Outcome, reward, and action-caused
observation records carry that ID so Python can reject forged or stale events.

## Deployment

The deployed runtime lives under `D:\PLWM-Luanti`:

```text
D:\PLWM-Luanti\engine\luanti-5.16.1-win64\bin\luanti.exe
D:\PLWM-Luanti\engine\luanti-5.16.1-win64\games\plwm_training
D:\PLWM-Luanti\world
D:\PLWM-Luanti\bridge
```

The server is bound to `127.0.0.1:30000`. The `plwm_training` mod is the only
trusted mod because it needs local file access to read the action queue and
append observations. It receives no HTTP permission.
