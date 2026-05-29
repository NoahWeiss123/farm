# farm-quest

Quest 3 VR client for FARM. Passthrough view + right controller pose
published to the FARM daemon over WiFi. Forked from the parent project's
`teleop_data_collector_quest`, trimmed for FARM-only use.

## What it does today

- Boots in **passthrough** so you see the real world through the headset
- Publishes `/q2r_right_hand_pose` (PoseStamped) over ROS-TCP to
  `tcp://<this-mac>:10000` at the headset's render rate (~72–90 Hz)
- FARM's ROS-TCP bridge routes the pose to the **ghost arm** target — the
  real UF850 is NOT commanded yet. Watching the digital arm follow the
  controller is the goal of this first cut.

## Defaults

- Host IP: `10.32.81.218` (this Mac's LAN address at fork time)
- Host port: `10000` (FARM ROS-TCP bridge)
- Package ID: `com.farm.quest`
- APK output: `artifacts/FarmQuest.apk`

The IP picker still works in-headset — hold `left A + left B` to open it
if the LAN address has shifted.

## Build + deploy

Requires Unity 6000.2.13f1 (the parent project's version — override with
`UNITY=...` for a different install) and Android Build Support installed
in Unity Hub.

```bash
cd farm-quest

# 1) Start the FARM daemon on this Mac
(cd ../farm-edge-agent && farm serve --backend xarm --arm-ip 192.168.1.220 --no-envelope)

# 2) Build + deploy + launch the APK on the paired Quest 3
./build.sh install run
```

The build script is two-pass — first pass adds the `ROS2` scripting define
the ROS-TCP-Connector needs, second pass produces the APK. Both passes
log to `logs/`.

## On the headset

1. Put on the Quest 3.
2. The app launches into passthrough — you see your room.
3. Watch the FARM dashboard in a browser on the Mac:
   `http://localhost:8787/`. As you rotate the right controller, the
   ghost (translucent cyan) arm follows the controller's orientation.

If the dashboard's status chip stays at "Waiting for joints…" the headset
isn't reaching the bridge — same WiFi network, no firewall on :10000.

## Folder layout

```
farm-quest/
├── build.sh                       # macOS → APK + adb install
├── UnityProject/
│   ├── Assets/
│   │   ├── Editor/Builder.cs      # headless APK build entrypoint
│   │   ├── Scripts/
│   │   │   ├── ConnectionConfig.cs    # IP/port + PlayerPrefs persistence
│   │   │   ├── Q2RPublisher.cs        # controller pose → ROS-TCP topics
│   │   │   ├── HUDController.cs       # in-headset status panel (legacy)
│   │   │   ├── ControllerPointer.cs   # raycast pointer (legacy)
│   │   │   ├── InteractableButton.cs  # button factory (legacy)
│   │   │   ├── PassthroughEnabler.cs  # FARM: make camera transparent
│   │   │   └── Messages/              # custom Quest2ROS msg defs
│   │   ├── Scenes/Main.unity      # XR rig + scripts
│   │   └── XR/Settings/           # OpenXR + Meta Quest config
│   ├── Packages/manifest.json     # ROS-TCP-Connector, OpenXR, Meta XR
│   └── ProjectSettings/
└── README.md
```

## What's not built yet (later milestones)

- Position-tracked ghost (today it tracks orientation only; position
  stays at the arm's last commanded TCP location)
- Re-anchor button (the existing Q2R's synthetic `button_upper` pulse is
  wired through but the FARM bridge doesn't honour it yet)
- Real arm motion driven by the Quest. This first cut is digital-only on
  purpose — verify the headset → bridge → ghost pipeline before letting
  the controller move a 7 kg arm.
