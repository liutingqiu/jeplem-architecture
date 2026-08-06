-- Local Luanti training arena. The only Python-controlled capabilities are
-- observe, reset, episode selection, and one-cell cardinal steps.
local core = core
local insecure = core.request_insecure_environment()

if not insecure then
    error("plwm_training requires secure.trusted_mods = plwm_training")
end

local bridge_root = core.settings:get("plwm_bridge_path")
if not bridge_root or bridge_root == "" then
    error("plwm_training requires plwm_bridge_path in the Luanti config")
end

local io_open = insecure.io.open
local actions_path = bridge_root .. "/actions.jsonl"
local observations_path = bridge_root .. "/observations.jsonl"
local arena_radius = 8
local tick = 0
local poll_elapsed = 0
local action_offset = 0
local episode_id = "luanti-session"
local task_seed = 0
local agent = {x = 0, z = 0, health = 20, energy = 20}
local goal = {x = 3, z = 3}
local arena_ready = false
local completed = false
local last_direction = nil
local last_blocked = false
local probe_recenter_available = false
local directional_steps = 0
local object_mode = false
local key = {x = 0, z = 0}
local decoy = {x = 0, z = 0}
local key_collected = false
local decoy_collected = false
local last_reward = 0
local hide_collection_facts = false
local charge_collected = false
local require_charge = false
local polarity_mode = false
local charge_mode = "key"
local failed = false
local multi_object_mode = false
local micro_navigation_mode = false
local delayed_micro_navigation_mode = false
local compact_delayed_micro_navigation_mode = false
local gated_compact_delayed_micro_navigation_mode = false
local long_gated_delayed_micro_navigation_mode = false
local paired_gated_delayed_micro_navigation_mode = false
local addressed_gated_delayed_micro_navigation_mode = false
local addressed_balanced_delayed_micro_navigation_mode = false
local addressed_balanced_2id_delayed_micro_navigation_mode = false
local paired_first_side = 1
local paired_second_side = 1
local paired_stage = 0
local addressed_sides = {1, 1, 1, 1}
local addressed_query = 1
local addressed_tiles = {"goal", "key", "decoy", "beacon"}
local candidate_positions = {}
local charge_choice = 1
local directions = {
    north = {x = 0, z = -1},
    south = {x = 0, z = 1},
    east = {x = 1, z = 0},
    west = {x = -1, z = 0},
}

core.register_node("plwm_training:floor", {
    description = "PLWM arena floor",
    tiles = {"blank.png^[colorize:#4B7F3D:255"},
    groups = {cracky = 3},
    is_ground_content = false,
})

core.register_node("plwm_training:wall", {
    description = "PLWM arena boundary",
    tiles = {"blank.png^[colorize:#485C6A:255"},
    groups = {cracky = 2},
    is_ground_content = false,
})

core.register_node("plwm_training:goal", {
    description = "PLWM training goal",
    tiles = {"blank.png^[colorize:#E6B83F:255"},
    light_source = 8,
    groups = {cracky = 1},
    is_ground_content = false,
})

core.register_node("plwm_training:key", {
    description = "PLWM task key",
    tiles = {"blank.png^[colorize:#3FA7D6:255"},
    light_source = 6,
    groups = {cracky = 1},
    is_ground_content = false,
})

core.register_node("plwm_training:decoy", {
    description = "PLWM task decoy",
    tiles = {"blank.png^[colorize:#C85A54:255"},
    light_source = 4,
    groups = {cracky = 1},
    is_ground_content = false,
})

core.register_node("plwm_training:beacon", {
    description = "PLWM retrieval beacon",
    tiles = {"blank.png^[colorize:#A76AD8:255"},
    light_source = 5,
    groups = {cracky = 1},
    is_ground_content = false,
})

local function next_seed(value)
    return (value * 48271) % 2147483647
end

local function addressed_mode_active()
    return addressed_gated_delayed_micro_navigation_mode
        or addressed_balanced_delayed_micro_navigation_mode
        or addressed_balanced_2id_delayed_micro_navigation_mode
end

local function gate_x()
    return next_seed(task_seed + 13) % 2 == 0 and -3 or 3
end

local function task_goal_x()
    -- The dedicated micro-navigation course is a public, visible-goal
    -- curriculum for validating reward-only learning plumbing.  It must not
    -- inherit the hidden gate or control-rule demands of later courses.
    if task_seed >= 8000000 and task_seed < 13000000 then
        return math.floor(next_seed(task_seed + 157) / 65536) % 2 == 0 and -1 or 1
    end
    if task_seed >= 6000000 and task_seed < 8000000 then
        return math.floor(next_seed(task_seed + 157) / 65536) % 2 == 0 and -2 or 2
    end
    -- Vision-only curriculum: the goal does not reveal the wall's gap.
    if task_seed >= 1400000 and task_seed < 1500000 then
        return math.floor(next_seed(task_seed + 71) / 65536) % 2 == 0 and -3 or 3
    end
    return gate_x()
end

local function task_goal_z()
    if task_seed >= 12000000 and task_seed < 13000000 then
        return -2
    end
    if task_seed >= 11000000 and task_seed < 12000000 then
        return -3
    end
    if task_seed >= 10000000 and task_seed < 11000000 then
        return -2
    end
    if task_seed >= 8000000 and task_seed < 10000000 then
        return -4
    end
    return micro_navigation_mode and -3 or 5
end

local function controls_inverted()
    if micro_navigation_mode then
        return false
    end
    local base_inverted = math.floor(next_seed(task_seed + 29) / 65536) % 2 == 1
    -- Dedicated high-seed tasks change the control mechanism after seven
    -- directional actions. This fact is intentionally absent from observation.
    if ((task_seed >= 1500000 and task_seed < 2000000) or task_seed >= 2500000) and directional_steps >= 7 then
        return not base_inverted
    end
    return base_inverted
end

local function actual_delta(direction)
    local requested = directions[direction]
    if controls_inverted() then
        return {x = -requested.x, z = -requested.z}
    end
    return requested
end

local function is_blocked(x, z)
    if addressed_mode_active() then
        if agent.x == 0 and agent.z < -2 then
            return not (z == agent.z + 1 and x == 0)
        elseif agent.x == 0 and agent.z == -2 then
            return not (z == -2 and (x == -1 or x == 1))
        end
    elseif paired_gated_delayed_micro_navigation_mode then
        -- Two public binary cues are visible only at the start. Five forced
        -- south moves hide both; then two ordered east/west decisions must
        -- replay the first and second cue without intermediate reward.
        if paired_stage == 0 then
            if agent.x == 0 and agent.z < -3 then
                return not (
                    (z == agent.z + 1 and x == 0)
                    or (agent.z == -8 and z == -3 and x == paired_first_side)
                    or (agent.z == -8 and z == -4 and x == paired_second_side)
                )
            elseif agent.x == 0 and agent.z == -3 then
                return not (z == -3 and (x == -1 or x == 1))
            elseif math.abs(agent.x) == 1 and agent.z == -3 then
                return not (z == -2 and x == agent.x)
            end
        elseif paired_stage == 1 and math.abs(agent.x) == 1 and agent.z == -3 then
            return not (z == -2 and x == agent.x)
        elseif paired_stage == 1 and math.abs(agent.x) == 1 and agent.z == -2 then
            return not (z == -2 and (x == agent.x - 1 or x == agent.x + 1))
        end
    elseif long_gated_delayed_micro_navigation_mode then
        -- The initial cue is visible from z=-7.  Five forced south moves
        -- remove it from view before the final east/west choice at z=-2.
        if agent.x == 0 and agent.z < -2 then
            return not (
                (z == agent.z + 1 and x == 0)
                or (agent.z == -7 and z == -2 and (x == -1 or x == 1))
            )
        elseif agent.x == 0 and agent.z == -2 then
            return not (z == -2 and (x == -1 or x == 1))
        end
    elseif gated_compact_delayed_micro_navigation_mode then
        -- A public two-action corridor: from the cue state only south is
        -- legal; after that transition only the two hidden-choice sides are
        -- legal.  The same walls are rendered in local vision.
        if agent.z == -5 then
            -- The two diagonal target cells remain visually open so the
            -- initial public frame can reveal the cue, while only south is
            -- adjacent to the agent and therefore action-legal.
            return not (z == -4 and (x == 0 or x == -1 or x == 1))
        elseif agent.x == 0 and agent.z == -4 then
            return not (z == -4 and (x == -1 or x == 1))
        end
    end
    return math.abs(x) >= arena_radius
        or math.abs(z) >= arena_radius
        -- In the delayed visible-goal course, the first public observation
        -- reveals the side but a corridor forces one forward action before
        -- the goal disappears from subsequent frames.
        or (delayed_micro_navigation_mode and z == -5 and x ~= 0)
        or (z == 0 and x ~= gate_x())
end

local function local_sensors()
    return {
        north = is_blocked(agent.x, agent.z - 1),
        south = is_blocked(agent.x, agent.z + 1),
        east = is_blocked(agent.x + 1, agent.z),
        west = is_blocked(agent.x - 1, agent.z),
    }
end

local function append_record(kind, modality, payload)
    local handle, error_message = io_open(observations_path, "ab")
    if not handle then
        core.log("error", "[plwm_training] unable to append observation: " .. tostring(error_message))
        return false
    end
    local record = {
        episode_id = episode_id,
        tick = tick,
        kind = kind,
        modality = modality,
        payload = payload,
    }
    handle:write(core.write_json(record), "\n")
    handle:close()
    return true
end

local function valid_request_id(value)
    return type(value) == "string"
        and #value >= 1
        and #value <= 128
        and value:match("^[A-Za-z0-9][A-Za-z0-9%._:%-]*$") ~= nil
end

local function write_outcome(name, status, request_id, extra)
    local payload = {name = name, status = status, request_id = request_id}
    if extra then
        for key, value in pairs(extra) do
            payload[key] = value
        end
    end
    append_record("outcome", "luanti-bridge", payload)
end

local function write_observation(caused_by)
    local sensors = local_sensors()
    append_record("observation", "luanti-state", {
        health = agent.health,
        energy = agent.energy,
        x = agent.x,
        z = agent.z,
        goal_x = goal.x,
        goal_z = goal.z,
        task_seed = task_seed,
        addressed_balanced = addressed_balanced_delayed_micro_navigation_mode,
        addressed_gated = addressed_gated_delayed_micro_navigation_mode,
        completed = completed,
        failed = failed,
        blocked_north = sensors.north,
        blocked_south = sensors.south,
        blocked_east = sensors.east,
        blocked_west = sensors.west,
        last_direction = last_direction,
        last_blocked = last_blocked,
        object_mode = object_mode,
        key_x = key.x,
        key_z = key.z,
        decoy_x = decoy.x,
        decoy_z = decoy.z,
        -- The composite memory curriculum preserves the causal reward record
        -- but does not repeat collection facts in each later observation.
        key_collected = (not hide_collection_facts) and key_collected or false,
        decoy_collected = (not hide_collection_facts) and decoy_collected or false,
        charge_collected = (not hide_collection_facts) and charge_collected or false,
        last_reward = last_reward,
        candidate_positions = candidate_positions,
        game_time = tick,
        caused_by = caused_by,
    })
end

local function write_vision(caused_by)
    local radius = 5
    local cells = {}
    for dz = -radius, radius do
        for dx = -radius, radius do
            local world_x = agent.x + dx
            local world_z = agent.z + dz
            local tile = "air"
            if addressed_mode_active() then
                if directional_steps == 0 then
                    -- All four identities and their independent left/right
                    -- placements are rendered in the initial public frame.
                    for index, side in ipairs(addressed_sides) do
                        if world_x == side * 2 and world_z == -8 + index then
                            tile = addressed_tiles[index]
                            break
                        end
                    end
                elseif agent.z == -2 and world_x == 0 and world_z == -4 then
                    -- The delayed public query identifies one cue but carries
                    -- no left/right answer information.
                    tile = addressed_tiles[addressed_query]
                end
            elseif paired_gated_delayed_micro_navigation_mode and directional_steps == 0 and world_x == key.x and world_z == key.z then
                tile = "key"
            elseif world_x == goal.x and world_z == goal.z and not (delayed_micro_navigation_mode and directional_steps >= 1) then
                tile = "goal"
            elseif multi_object_mode then
                for index, candidate in ipairs(candidate_positions) do
                    if world_x == candidate.x and world_z == candidate.z then
                        tile = index == 1 and "key" or "decoy"
                        break
                    end
                end
            elseif object_mode and (not key_collected or hide_collection_facts) and world_x == key.x and world_z == key.z then
                tile = "key"
            elseif object_mode and (not decoy_collected or hide_collection_facts) and world_x == decoy.x and world_z == decoy.z then
                tile = "decoy"
            end
            -- Courses may expose a non-adjacent cue while the movement rule
            -- keeps it unreachable. Render that public object before using
            -- the same local wall contract that masks legal adjacent actions.
            if tile == "air" and is_blocked(world_x, world_z) then
                tile = "wall"
            end
            if world_x == agent.x and world_z == agent.z then
                tile = "agent"
            end
            table.insert(cells, tile)
        end
    end
    append_record("observation", "luanti-vision", {
        width = radius * 2 + 1,
        height = radius * 2 + 1,
        encoding = "tile-grid-v1",
        cells = cells,
        caused_by = caused_by,
    })
end

local function build_arena()
    for x = -arena_radius, arena_radius do
        for z = -arena_radius, arena_radius do
            local node_name = "plwm_training:floor"
            if math.abs(x) == arena_radius or math.abs(z) == arena_radius then
                node_name = "plwm_training:wall"
            end
            core.set_node({x = x, y = 0, z = z}, {name = node_name})
            core.set_node({x = x, y = 1, z = z}, {name = "air"})
        end
    end
    for x = -arena_radius + 1, arena_radius - 1 do
        if x ~= gate_x() then
            core.set_node({x = x, y = 1, z = 0}, {name = "plwm_training:wall"})
        end
    end
    if addressed_mode_active() then
        local count = 2
        if addressed_balanced_delayed_micro_navigation_mode or addressed_gated_delayed_micro_navigation_mode then
            count = 4
        end
        for index = 1, count do
            core.set_node({x = 0, y = 1, z = -8 + index}, {name = "air"})
        end
        for index, side in ipairs(addressed_sides) do
            core.set_node({x = side * 2, y = 1, z = -8 + index}, {name = "plwm_training:" .. addressed_tiles[index]})
        end
        core.set_node({x = 0, y = 1, z = -4}, {name = "plwm_training:" .. addressed_tiles[addressed_query]})
    else
        core.set_node({x = goal.x, y = 1, z = goal.z}, {name = "plwm_training:goal"})
    end
    if paired_gated_delayed_micro_navigation_mode then
        core.set_node({x = key.x, y = 1, z = key.z}, {name = "plwm_training:key"})
    elseif multi_object_mode then
        for index, candidate in ipairs(candidate_positions) do
            core.set_node({x = candidate.x, y = 1, z = candidate.z}, {name = index == 1 and "plwm_training:key" or "plwm_training:decoy"})
        end
    elseif object_mode then
        core.set_node({x = key.x, y = 1, z = key.z}, {name = "plwm_training:key"})
        core.set_node({x = decoy.x, y = 1, z = decoy.z}, {name = "plwm_training:decoy"})
    end
end

local function valid_task_seed(value)
    return type(value) == "number"
        and value == math.floor(value)
        and value >= 0
        and value < 2147483647
end

local function reset_agent(request_id)
    agent = {x = 0, z = -5, health = 20, energy = 20}
    long_gated_delayed_micro_navigation_mode = task_seed >= 10000000 and task_seed < 11000000
    paired_gated_delayed_micro_navigation_mode = task_seed >= 11000000 and task_seed < 12000000
    addressed_gated_delayed_micro_navigation_mode = task_seed >= 12000000 and task_seed < 13000000
    addressed_balanced_delayed_micro_navigation_mode = task_seed >= 16000000 and task_seed < 17000000
    addressed_balanced_2id_delayed_micro_navigation_mode = task_seed >= 18000000 and task_seed < 19000000
    if addressed_balanced_delayed_micro_navigation_mode or addressed_balanced_2id_delayed_micro_navigation_mode then
        minetest.log("action", "[plwm] balanced seed " .. task_seed)
    end
    gated_compact_delayed_micro_navigation_mode = task_seed >= 9000000 and task_seed < 10000000
    compact_delayed_micro_navigation_mode = (task_seed >= 8000000 and task_seed < 10000000) or long_gated_delayed_micro_navigation_mode
    delayed_micro_navigation_mode = task_seed >= 7000000 and task_seed < 13000000
    micro_navigation_mode = task_seed >= 6000000 and task_seed < 13000000
    if paired_gated_delayed_micro_navigation_mode then
        agent.z = -8
    elseif addressed_mode_active() then
        agent.z = -7
    elseif long_gated_delayed_micro_navigation_mode then
        agent.z = -7
    end
    goal = {x = task_goal_x(), z = task_goal_z()}
    object_mode = task_seed >= 2000000 and not micro_navigation_mode and not addressed_mode_active()
    multi_object_mode = task_seed >= 5000000 and not micro_navigation_mode and not addressed_mode_active()
    -- The richer course places the key on the opposite side of the lower
    -- arena. Reaching the goal to charge no longer collects the key en route.
    paired_first_side = goal.x
    paired_second_side = math.floor(next_seed(task_seed + 181) / 65536) % 2 == 0 and -1 or 1
    paired_stage = 0
    addressed_sides = {}
    for index = 1, 4 do
        addressed_sides[index] = math.floor(next_seed(task_seed + 181 + index * 37) / 65536) % 2 == 0 and -1 or 1
    end
    addressed_query = math.floor(next_seed(task_seed + 401) / 65536) % 4 + 1
    if addressed_balanced_delayed_micro_navigation_mode or addressed_balanced_2id_delayed_micro_navigation_mode then
        -- A seed-range counterbalanced addressed variant.  The query index is
        -- the seed offset modulo 4 (or modulo 2 for the two-identity
        -- difficulty-reduction variant) and every identity side is the parity
        -- of (floor(offset / count) + identity).  Over every contiguous
        -- window whose length is a multiple of four (respectively two), each
        -- identity is left/right exactly half the time and each identity is
        -- queried an equal number of times, so the identity->side marginal
        -- carries no exploitable frequency signal.  This is the single change
        -- that separates an intrinsic side bias from the delayed memory
        -- signal in the object-token task.
        local count = 2
        if addressed_balanced_delayed_micro_navigation_mode then
            count = 4
        end
        local offset = task_seed - 16000000
        if addressed_balanced_2id_delayed_micro_navigation_mode then
            offset = task_seed - 18000000
        end
        addressed_sides = {}
        for index = 1, count do
            addressed_sides[index] = (math.floor(offset / count) + index) % 2 == 0 and 1 or -1
        end
        addressed_query = offset % count + 1
    end
    if paired_gated_delayed_micro_navigation_mode then
        candidate_positions = {}
        key = {x = paired_second_side, z = -4}
        decoy = {x = 0, z = 0}
    elseif multi_object_mode then
        -- Four public, indexed candidates share the lower arena.  The only
        -- answer is the earlier charge outcome; it is cleared on recenter.
        candidate_positions = {
            -- Every candidate occupies the same non-start row. The teacher
            -- can therefore reach each indexed choice without crossing a
            -- different candidate, which would turn path geometry into an
            -- unintended answer or premature failure.
            {index = 1, x = -5, z = -3},
            {index = 2, x = -2, z = -3},
            {index = 3, x = 2, z = -3},
            {index = 4, x = 5, z = -3},
        }
        key = candidate_positions[1]
        decoy = candidate_positions[2]
    elseif task_seed >= 4000000 then
        candidate_positions = {}
        key = {x = -goal.x, z = -3}
        decoy = {x = -goal.x, z = -5}
    elseif task_seed >= 3000000 then
        candidate_positions = {}
        key = {x = -goal.x, z = -3}
        decoy = {x = -goal.x, z = 3}
    else
        candidate_positions = {}
        key = {x = goal.x, z = -3}
        decoy = {x = -goal.x, z = -2}
    end
    key_collected = false
    decoy_collected = false
    last_reward = 0
    hide_collection_facts = task_seed >= 2700000 and not micro_navigation_mode and not addressed_mode_active()
    charge_collected = false
    require_charge = task_seed >= 3000000 and not micro_navigation_mode and not addressed_mode_active()
    polarity_mode = task_seed >= 4000000 and task_seed < 5000000 and not micro_navigation_mode and not addressed_mode_active()
    charge_mode = math.floor(next_seed(task_seed + 83) / 65536) % 2 == 0 and "key" or "decoy"
    charge_choice = math.floor(next_seed(task_seed + 97) / 65536) % 4 + 1
    completed = false
    failed = false
    last_direction = nil
    last_blocked = false
    probe_recenter_available = false
    directional_steps = 0
    build_arena()
end

local function ensure_arena_ready(name, request_id)
    if arena_ready then
        return true
    end
    write_outcome(name, "rejected", request_id, {reason = "world_not_ready"})
    return false
end

local function step_agent(direction, request_id)
    local delta = directions[direction]
    if not delta then
        write_outcome("step", "rejected", request_id, {reason = "invalid_direction"})
        return
    end
    if completed or failed then
        write_outcome("step", "rejected", request_id, {reason = "episode_complete"})
        return
    end
    local is_probe = last_direction == nil
    local delta = actual_delta(direction)
    local next_x = agent.x + delta.x
    local next_z = agent.z + delta.z
    local blocked = is_blocked(next_x, next_z)
    last_direction = direction
    last_blocked = blocked
    local reward = -0.01
    if not blocked then
        agent.x = next_x
        agent.z = next_z
        agent.energy = math.max(0, agent.energy - 0.05)
    end
    if multi_object_mode and charge_collected and not key_collected then
        for index, candidate in ipairs(candidate_positions) do
            if agent.x == candidate.x and agent.z == candidate.z then
                if index == charge_choice then
                    key_collected = true
                    reward = 0.2
                else
                    failed = true
                    reward = -1.0
                end
                break
            end
        end
    elseif object_mode and agent.x == key.x and agent.z == key.z and not key_collected then
        if not require_charge or charge_collected then
            if not polarity_mode or charge_mode == "key" then
                key_collected = true
                reward = 0.2
            else
                decoy_collected = true
                reward = -0.05
            end
        end
    elseif object_mode and agent.x == decoy.x and agent.z == decoy.z and not decoy_collected then
        if not require_charge or charge_collected then
            if polarity_mode and charge_mode == "decoy" then
                key_collected = true
                decoy_collected = true
                reward = 0.2
            else
                decoy_collected = true
                reward = -0.05
            end
        end
    end
    if addressed_mode_active() and agent.z == -2 and math.abs(agent.x) == 1 then
        if agent.x == addressed_sides[addressed_query] then
            completed = true
            reward = 1.0
        else
            failed = true
            reward = -1.0
        end
    elseif paired_gated_delayed_micro_navigation_mode then
        if paired_stage == 0 and agent.z == -3 and math.abs(agent.x) == 1 then
            if agent.x == paired_first_side then
                paired_stage = 1
            else
                failed = true
                reward = -1.0
            end
        elseif paired_stage == 1 and agent.z == -2 and delta.z == 0 and math.abs(delta.x) == 1 then
            if delta.x == paired_second_side then
                completed = true
                reward = 1.0
            else
                failed = true
                reward = -1.0
            end
        end
    elseif compact_delayed_micro_navigation_mode and agent.z == goal.z and agent.x == -goal.x then
        -- A delayed binary decision cannot be solved by visiting both sides.
        failed = true
        reward = -1.0
    end
    if not paired_gated_delayed_micro_navigation_mode and not addressed_mode_active() and agent.x == goal.x and agent.z == goal.z then
        if require_charge and not charge_collected then
            charge_collected = true
            if multi_object_mode then
                reward = 0.14 + 0.01 * charge_choice
            else
                reward = polarity_mode and (charge_mode == "key" and 0.15 or 0.16) or 0.15
            end
        elseif not object_mode or (key_collected and (not require_charge or charge_collected)) then
            reward = 1.0
            completed = true
        end
    end
    last_reward = reward
    if is_probe then
        probe_recenter_available = true
    end
    directional_steps = directional_steps + 1
    append_record("reward", "luanti-reward", {value = reward, blocked = blocked, terminal = completed or failed, success = completed, caused_by = request_id})
    write_outcome("step", "accepted", request_id, {blocked = blocked, reward = reward, terminal = completed or failed, success = completed})
    write_observation(request_id)
    write_vision(request_id)
end

local function recenter_agent(request_id)
    if not probe_recenter_available and not (hide_collection_facts and (key_collected or charge_collected)) then
        write_outcome("recenter", "rejected", request_id, {reason = "probe_not_available"})
        return
    end
    agent.x = 0
    agent.z = -5
    probe_recenter_available = false
    if hide_collection_facts then
        last_reward = 0
        last_direction = nil
        last_blocked = false
    end
    write_outcome("recenter", "accepted", request_id)
    write_observation(request_id)
    write_vision(request_id)
end

local function handle_action(action)
    if type(action) ~= "table" or not valid_request_id(action.request_id) then
        write_outcome("invalid", "rejected", nil, {reason = "invalid_request"})
        return
    end
    if action.name == "observe" then
        write_outcome("observe", "accepted", action.request_id)
        write_observation(action.request_id)
        write_vision(action.request_id)
    elseif action.name == "reset_episode" then
        if not ensure_arena_ready("reset_episode", action.request_id) then
            return
        end
        reset_agent(action.request_id)
        write_outcome("reset_episode", "accepted", action.request_id)
        write_observation(action.request_id)
        write_vision(action.request_id)
    elseif action.name == "set_episode" and type(action.episode_id) == "string" and action.episode_id ~= "" and (action.task_seed == nil or valid_task_seed(action.task_seed)) then
        episode_id = action.episode_id
        task_seed = action.task_seed or 0
        write_outcome("set_episode", "accepted", action.request_id, {task_seed = task_seed})
    elseif action.name == "step" then
        if not ensure_arena_ready("step", action.request_id) then
            return
        end
        step_agent(action.direction, action.request_id)
    elseif action.name == "recenter" then
        if not ensure_arena_ready("recenter", action.request_id) then
            return
        end
        recenter_agent(action.request_id)
    else
        write_outcome(tostring(action.name), "rejected", action.request_id, {reason = "not_allowlisted"})
    end
end

local function count_existing_actions()
    local handle = io_open(actions_path, "rb")
    if not handle then
        return 0
    end
    local count = 0
    for _ in handle:lines() do
        count = count + 1
    end
    handle:close()
    return count
end

local function poll_actions()
    local handle = io_open(actions_path, "rb")
    if not handle then
        return
    end
    local line_number = 0
    for line in handle:lines() do
        line_number = line_number + 1
        if line_number > action_offset and line ~= "" then
            handle_action(core.parse_json(line))
        end
    end
    handle:close()
    action_offset = line_number
end

action_offset = count_existing_actions()

core.after(0, function()
    core.emerge_area(
        {x = -arena_radius, y = 0, z = -arena_radius},
        {x = arena_radius, y = 1, z = arena_radius},
        function(_, _, calls_remaining)
            if calls_remaining == 0 then
                build_arena()
                arena_ready = true
                append_record("checkpoint", "luanti-bridge", {state = "started", engine = core.get_version().string})
            end
        end
    )
end)

core.register_globalstep(function(delta)
    tick = tick + 1
    poll_elapsed = poll_elapsed + delta
    if poll_elapsed >= 0.1 then
        poll_elapsed = 0
        poll_actions()
    end
end)

core.register_on_shutdown(function()
    append_record("checkpoint", "luanti-bridge", {state = "stopped"})
end)
