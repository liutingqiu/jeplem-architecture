# PLWM Minecraft Training Bridge

This Fabric 1.21 mod provides a deliberately narrow, local-only bridge between
a single-player Minecraft world and the PLWM research runtime. It does not open
a network port and does not accept arbitrary Minecraft commands.

## Build

Use JDK 21 and run:

```text
gradlew.bat build
```

Copy `build/libs/plwm-training-0.1.0.jar` to the world's Fabric `mods` folder
along with the matching Fabric API. The mod creates `plwm-training` under the
Minecraft game directory after a server starts.

## Protocol boundary

The mod appends raw JSONL observations to `observations.jsonl` and consumes
newline-delimited action objects from `actions.jsonl`. Each action requires a
unique ASCII `request_id`; its terminal outcome includes the same ID. The only
accepted action names are `observe`, `set_episode`, and `reset_episode`.

`reset_episode` only teleports online players to the overworld shared spawn.
The resulting player observations carry that action's `request_id` as
`caused_by`, preserving the action-to-state causal link in the raw trace.
Movement, looking, block interaction, inventory changes, chat, commands, and
network access are intentionally unavailable. Historical action lines are
ignored after a server restart, so an action can be executed at most once by a
running bridge instance.

Python code in the parent project is responsible for validating, hash-chaining,
and converting raw records into the PLWM event protocol. The offline contract
test is `python experiments/minecraft_bridge_contract.py`.
