// AR-Foundation-based passthrough. When this is in place, controllers
// enumerate via legacy InputDevices on Quest 3 (presumably because
// AR Foundation's ARSession kicks the OpenXR session into the right
// state). Removing it broke controller detection in this session; put
// back as the last config delta to restore.

using UnityEngine;
using UnityEngine.XR.ARFoundation;

namespace TeleopDataCollector
{
    public class PassthroughEnabler : MonoBehaviour
    {
        bool applied;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Spawn()
        {
            var go = new GameObject("FARM_PassthroughEnabler");
            DontDestroyOnLoad(go);
            go.AddComponent<PassthroughEnabler>();
        }

        void Update()
        {
            if (applied) return;
            var cam = Camera.main;
            if (cam == null) return;

            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0f, 0f, 0f, 0f);

            if (Object.FindFirstObjectByType<ARSession>() == null)
            {
                var sessGO = new GameObject("FARM_ARSession");
                DontDestroyOnLoad(sessGO);
                sessGO.AddComponent<ARSession>();
            }
            if (cam.GetComponent<ARCameraManager>() == null)
                cam.gameObject.AddComponent<ARCameraManager>();
            if (cam.GetComponent<ARCameraBackground>() == null)
                cam.gameObject.AddComponent<ARCameraBackground>();

            applied = true;
            Debug.Log("[FARM] passthrough + ARSession applied");
        }
    }
}
