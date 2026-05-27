// Force the XR tracking origin into Floor mode so world (0,0,0) is on
// the floor at the user's feet — without this, controllers report poses
// way below the user (their world Y is the floor, but Camera.main is at
// roughly head height, putting the relative offset ~1.6 m down).
//
// Self-bootstraps on scene load so we don't have to edit Main.unity.

using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

namespace TeleopDataCollector
{
    public class TrackingOriginSetter : MonoBehaviour
    {
        bool applied;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Spawn()
        {
            var go = new GameObject("FARM_TrackingOriginSetter");
            DontDestroyOnLoad(go);
            go.AddComponent<TrackingOriginSetter>();
        }

        float lastLogT;

        void Update()
        {
            if (applied) return;
            var subs = new List<XRInputSubsystem>();
            SubsystemManager.GetInstances(subs);
            foreach (var s in subs)
            {
                if (s == null || !s.running) continue;
                // Try Floor first — poses then report metres above the
                // physical floor (what the bridge / dashboard expect).
                // Fall back to Unbounded if Floor isn't supported.
                bool ok = s.TrySetTrackingOriginMode(TrackingOriginModeFlags.Floor);
                var modeNow = s.GetTrackingOriginMode();
                if (ok && modeNow == TrackingOriginModeFlags.Floor)
                {
                    Debug.Log($"[FARM] tracking origin → Floor (mode={modeNow})");
                    applied = true;
                    return;
                }
                if (s.TrySetTrackingOriginMode(TrackingOriginModeFlags.Unbounded))
                {
                    Debug.Log($"[FARM] tracking origin → Unbounded (mode={s.GetTrackingOriginMode()})");
                    applied = true;
                    return;
                }
                if (Time.unscaledTime - lastLogT > 2f)
                {
                    Debug.LogWarning($"[FARM] tracking origin set failed on {s} (current mode={modeNow})");
                    lastLogT = Time.unscaledTime;
                }
            }
        }
    }
}
