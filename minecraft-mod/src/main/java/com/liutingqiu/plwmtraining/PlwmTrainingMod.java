package com.liutingqiu.plwmtraining;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class PlwmTrainingMod implements ModInitializer {
    public static final String MOD_ID = "plwm_training";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    @Override
    public void onInitialize() {
        LocalTraceBridge bridge = new LocalTraceBridge();
        ServerLifecycleEvents.SERVER_STARTED.register(bridge::onServerStarted);
        ServerLifecycleEvents.SERVER_STOPPING.register(bridge::onServerStopping);
        ServerTickEvents.END_SERVER_TICK.register(bridge::onServerTick);
        LOGGER.info("PLWM training bridge initialized; local file bridge is disabled until a world starts.");
    }
}
