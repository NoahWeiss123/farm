#!/usr/bin/env bash
# Build + (optionally) install FARM Quest to a paired Quest 3.
#
# Usage:
#   ./build.sh              # build APK only
#   ./build.sh install      # build + adb install -r
#   ./build.sh install run  # build + install + launch on-headset
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
UNITY="${UNITY:-/Applications/Unity/Hub/Editor/6000.2.13f1/Unity.app/Contents/MacOS/Unity}"
PKG_ID="com.farm.quest"
APK="$ROOT/artifacts/FarmQuest.apk"
LOG="$ROOT/logs/build-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$ROOT/logs" "$ROOT/artifacts"

if [[ ! -x "$UNITY" ]]; then
  echo "error: Unity not found at $UNITY (override with UNITY=...)" >&2
  exit 1
fi

# Two-pass build:
#   pass 1: ensure the ROS2 scripting-define is set on the Android target.
#           Required because RO-S-TCP-Connector uses `#if ROS2`. If the define
#           isn't set yet the connector compiles in ROS1 wire format and the
#           host quest_teleop_node will mis-deserialize.
#   pass 2: actual APK build, now with ROS2 active.
LOG1="${LOG%.log}-pass1.log"
LOG2="${LOG%.log}-pass2.log"

# Skip pass 1 if the ROS2 define is already in ProjectSettings.asset — saves
# ~30-60 s of Unity batch-mode boot on every incremental build.
PROJ_SETTINGS="$ROOT/UnityProject/ProjectSettings/ProjectSettings.asset"
if grep -q "scriptingDefineSymbols.*ROS2\|;ROS2$\|;ROS2;\|: ROS2" "$PROJ_SETTINGS" 2>/dev/null; then
  echo "[build/p1] ROS2 define already set; skipping pass 1"
else
  echo "[build/p1] ensure ROS2 define -> $LOG1"
  "$UNITY" -batchmode -nographics -quit \
    -projectPath "$ROOT/UnityProject" \
    -buildTarget Android \
    -executeMethod TeleopDataCollector.Builder.EnsureROS2Define \
    -logFile "$LOG1"
fi

echo "[build/p2] APK build -> $LOG2"
"$UNITY" -batchmode -nographics -quit \
  -projectPath "$ROOT/UnityProject" \
  -buildTarget Android \
  -executeMethod TeleopDataCollector.Builder.BuildAndroid \
  -logFile "$LOG2"

if [[ ! -f "$APK" ]]; then
  echo "[build] FAILED -- no APK produced. tail of pass2 log:" >&2
  tail -n 80 "$LOG2" >&2
  exit 1
fi
echo "[build] OK -> $APK ($(du -h "$APK" | cut -f1))"

case "${1-}" in
  install*|i)
    DEVS=$(adb devices | awk 'NR>1 && $2=="device" {print $1}')
    if [[ -z "$DEVS" ]]; then
      echo "[install] no adb device -- plug the Quest in + enable dev mode" >&2
      exit 1
    fi
    # pm clear wipes the prior app's PlayerPrefs (which `adb install -r` would otherwise
    # preserve). Without this, a baked-in IP change in source never reaches the runtime
    # because the singleton reads the stale saved value.
    echo "[install] adb shell pm clear $PKG_ID (wipes PlayerPrefs)"
    adb shell pm clear "$PKG_ID" 2>&1 | sed 's/^/  /'
    echo "[install] adb install -r $APK"
    adb install -r "$APK"
    if [[ "${2-}" == "run" ]]; then
      echo "[install] launching $PKG_ID on headset"
      # Unity 6 uses UnityPlayerGameActivity (GameActivity base class).
      adb shell am start -n "$PKG_ID/com.unity3d.player.UnityPlayerGameActivity"
    fi
    ;;
esac
