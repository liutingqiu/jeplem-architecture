package com.liutingqiu.plwmtraining;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.core.BlockPos;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.List;
import java.util.regex.Pattern;

/**
 * A local, append-only bridge. It exposes no socket and never executes raw
 * Minecraft commands. Python normalizes these raw records into the hash-chained
 * PLWM event protocol before training or replay.
 */
final class LocalTraceBridge {
    private static final Gson GSON = new Gson();
    private static final int OBSERVATION_INTERVAL_TICKS = 20;
    private static final Pattern REQUEST_ID_PATTERN = Pattern.compile("[A-Za-z0-9][A-Za-z0-9._:-]{0,127}");
    private final Path root = FabricLoader.getInstance().getGameDir().resolve("plwm-training");
    private Path observationsPath;
    private Path actionsPath;
    private long actionOffset = 0;
    private long tick = 0;
    private String episodeId = "minecraft-session";

    void onServerStarted(MinecraftServer server) {
        try {
            Files.createDirectories(root);
            observationsPath = root.resolve("observations.jsonl");
            actionsPath = root.resolve("actions.jsonl");
            // Historical records describe completed work. A server restart must
            // only consume actions appended after this bridge has started.
            actionOffset = Files.exists(actionsPath)
                ? Files.readAllLines(actionsPath, StandardCharsets.UTF_8).size()
                : 0;
            writeRecord("checkpoint", "bridge", metadata(server));
            PlwmTrainingMod.LOGGER.info("PLWM local bridge ready at {}", root);
        } catch (IOException error) {
            throw new IllegalStateException("Unable to initialize PLWM training bridge", error);
        }
    }

    void onServerStopping(MinecraftServer server) {
        if (observationsPath != null) {
            writeRecord("checkpoint", "bridge", metadata(server));
        }
    }

    void onServerTick(MinecraftServer server) {
        tick++;
        if (observationsPath == null) {
            return;
        }
        pollActions(server);
        if (tick % OBSERVATION_INTERVAL_TICKS == 0) {
            for (ServerPlayer player : server.getPlayerList().getPlayers()) {
                writeObservation(server, player);
            }
        }
    }

    private void pollActions(MinecraftServer server) {
        if (!Files.exists(actionsPath)) {
            return;
        }
        try {
            List<String> lines = Files.readAllLines(actionsPath, StandardCharsets.UTF_8);
            while (actionOffset < lines.size()) {
                handleAction(server, lines.get((int) actionOffset));
                actionOffset++;
            }
        } catch (IOException error) {
            PlwmTrainingMod.LOGGER.warn("Could not read PLWM actions file", error);
        }
    }

    private void handleAction(MinecraftServer server, String line) {
        try {
            JsonObject action = JsonParser.parseString(line).getAsJsonObject();
            String name = action.get("name").getAsString();
            String requestId = action.get("request_id").getAsString();
            if (!REQUEST_ID_PATTERN.matcher(requestId).matches()) {
                throw new IllegalArgumentException("invalid request ID");
            }
            // Deliberate allowlist. Movement/interactions require a later client-side
            // capability layer; raw command execution is never permitted here.
            if ("observe".equals(name)) {
                for (ServerPlayer player : server.getPlayerList().getPlayers()) {
                    writeObservation(server, player);
                }
                writeRecord("outcome", "bridge", actionOutcome(name, "accepted", requestId));
            } else if ("reset_episode".equals(name)) {
                resetPlayers(server, requestId);
                writeRecord("outcome", "bridge", actionOutcome(name, "accepted", requestId));
            } else if ("set_episode".equals(name) && action.has("episode_id")) {
                episodeId = action.get("episode_id").getAsString();
                writeRecord("outcome", "bridge", actionOutcome(name, "accepted", requestId));
            } else {
                writeRecord("outcome", "bridge", actionOutcome(name, "rejected", requestId));
            }
        } catch (RuntimeException error) {
            writeRecord("outcome", "bridge", actionOutcome("invalid", "rejected", null));
        }
    }

    private void resetPlayers(MinecraftServer server, String requestId) {
        ServerLevel level = server.overworld();
        BlockPos spawn = level.getSharedSpawnPos();
        for (ServerPlayer player : server.getPlayerList().getPlayers()) {
            player.teleportTo(level, spawn.getX() + 0.5, spawn.getY(), spawn.getZ() + 0.5, player.getYRot(), player.getXRot());
            writeObservation(server, player, requestId);
        }
    }

    private void writeObservation(MinecraftServer server, ServerPlayer player) {
        writeObservation(server, player, null);
    }

    private void writeObservation(MinecraftServer server, ServerPlayer player, String causedBy) {
        JsonObject payload = new JsonObject();
        payload.addProperty("player_id", player.getStringUUID());
        payload.addProperty("x", player.getX());
        payload.addProperty("y", player.getY());
        payload.addProperty("z", player.getZ());
        payload.addProperty("health", player.getHealth());
        payload.addProperty("food", player.getFoodData().getFoodLevel());
        payload.addProperty("game_time", server.overworld().getGameTime());
        if (causedBy != null) {
            payload.addProperty("caused_by", causedBy);
        }
        writeRecord("observation", "minecraft-state", payload);
    }

    private JsonObject metadata(MinecraftServer server) {
        JsonObject payload = new JsonObject();
        payload.addProperty("episode_id", episodeId);
        payload.addProperty("game_time", server.overworld().getGameTime());
        payload.addProperty("wall_time", Instant.now().toString());
        return payload;
    }

    private JsonObject actionOutcome(String name, String status, String requestId) {
        JsonObject payload = new JsonObject();
        payload.addProperty("name", name);
        payload.addProperty("status", status);
        if (requestId != null) {
            payload.addProperty("request_id", requestId);
        }
        return payload;
    }

    private void writeRecord(String kind, String modality, JsonObject payload) {
        JsonObject record = new JsonObject();
        record.addProperty("episode_id", episodeId);
        record.addProperty("tick", tick);
        record.addProperty("kind", kind);
        record.addProperty("modality", modality);
        record.add("payload", payload);
        try {
            Files.writeString(observationsPath, GSON.toJson(record) + System.lineSeparator(), StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException error) {
            PlwmTrainingMod.LOGGER.warn("Could not append PLWM bridge record", error);
        }
    }
}
