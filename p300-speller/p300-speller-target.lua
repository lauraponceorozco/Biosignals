-- This Lua script generates target stimulations for the P300 visualisation
-- box based on the matrix of letters / numbers a P300 speller has
--
-- Modified for 3×3 speller with digits 1 to 9

grid =
{
    { '1', '2', '3' },
    { '4', '5', '6' },
    { '7', '8', '9' }
}

index = 0

function get_location(c)
    for i = 1, #grid do
        for j = 1, #grid[i] do
            if grid[i][j] == c then
                return i, j
            end
        end
    end
    return 0, 0
end

-- this function is called when the box is initialized
function initialize(box)
    dofile(box:get_config("${Path_Data}") .. "/plugins/stimulation/lua-stimulator-stim-codes.lua")

    math.randomseed(os.time())
    target = box:get_setting(2)
    row_base = _G[box:get_setting(3)]
    col_base = _G[box:get_setting(4)]
    delay = box:get_setting(5)

    if target == "" then
        for i = 1, 1000 do
            a = math.random(1, #grid)
            b = math.random(1, #grid[1])
            target = target .. grid[a][b]
        end
    end
end

function uninitialize(box)
end

function process(box)
    while box:keep_processing() do
        t = box:get_current_time()

        for stimulation = 1, box:get_stimulation_count(1) do
            stimulation_id, stimulation_time, stimulation_duration = box:get_stimulation(1, 1)

            if stimulation_id == OVTK_StimulationId_RestStart then
                index = index + 1
                r, c = get_location(string.sub(target, index, index))
                box:send_stimulation(1, row_base + r - 1, t + delay, 0)
                box:send_stimulation(1, col_base + c - 1, t + delay, 0)

            elseif stimulation_id == OVTK_StimulationId_ExperimentStop then
                box:send_stimulation(1, OVTK_StimulationId_Train, t + delay + 1, 0)
            end

            box:remove_stimulation(1, 1)
        end

        box:sleep()
    end
end
