// Drive the Main Camera's localTransform from the HMD pose every frame.
//
// XROrigin + plain Camera under Unity 6 OpenXR doesn't auto-track the
// camera transform on Quest — only the per-eye view matrices do. So a
// world-anchored object computed against Camera.main.transform.position
// reads as head-locked because the camera GameObject sits at the XROrigin
// offset and never updates with the headset. We bridge that gap by
// reading the HMD pose ourselves.
//
// Two pose sources are tried in order:
//   1. UnityEngine.XR.InputTracking.GetLocalPosition(XRNode.CenterEye) —
//      the legacy XR API. Works on Quest+OpenXR where InputDevices does
//      not populate CommonUsages.centerEyePosition.
//   2. InputDevices.GetDeviceAtXRNode(XRNode.CenterEye/Head) — fallback.
//
// Runs in LateUpdate + BeforeRender so the camera transform reflects the
// freshest possible pose before any culling/rendering happens.

using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

namespace TeleopDataCollector
{
    [DefaultExecutionOrder(-30000)]
    public class CameraPoseDriver : MonoBehaviour
    {
        InputDevice fallbackHead;
        float lastDiagT;

        void OnEnable()  { Application.onBeforeRender += Apply; }
        void OnDisable() { Application.onBeforeRender -= Apply; }
        void Update()      => Apply();
        void LateUpdate()  => Apply();

        void Apply()
        {
            Vector3 pos = Vector3.zero;
            Quaternion rot = Quaternion.identity;
            bool gotPos = false, gotRot = false;
            string src = "none";

            // 1) Legacy XR.InputTracking — fires on OpenXR even when
            //    InputDevices.CommonUsages stays at defaults.
            #pragma warning disable CS0618
            var trackedPos = InputTracking.GetLocalPosition(XRNode.CenterEye);
            var trackedRot = InputTracking.GetLocalRotation(XRNode.CenterEye);
            #pragma warning restore CS0618
            if (trackedPos != Vector3.zero) { pos = trackedPos; gotPos = true; src = "InputTracking"; }
            if (trackedRot != Quaternion.identity) { rot = trackedRot; gotRot = true; }

            // 2) InputDevices fallback.
            if (!gotPos || !gotRot)
            {
                if (!fallbackHead.isValid)
                {
                    fallbackHead = InputDevices.GetDeviceAtXRNode(XRNode.CenterEye);
                    if (!fallbackHead.isValid)
                        fallbackHead = InputDevices.GetDeviceAtXRNode(XRNode.Head);
                    if (!fallbackHead.isValid)
                    {
                        var devs = new List<InputDevice>();
                        InputDevices.GetDevicesWithCharacteristics(
                            InputDeviceCharacteristics.HeadMounted, devs);
                        if (devs.Count > 0) fallbackHead = devs[0];
                    }
                }
                if (fallbackHead.isValid)
                {
                    if (!gotPos &&
                        fallbackHead.TryGetFeatureValue(CommonUsages.centerEyePosition, out var p) &&
                        p != Vector3.zero)
                    { pos = p; gotPos = true; src = "InputDevices.centerEye"; }
                    if (!gotRot &&
                        fallbackHead.TryGetFeatureValue(CommonUsages.centerEyeRotation, out var r) &&
                        r != default)
                    { rot = r; gotRot = true; }
                }
            }

            if (gotPos) transform.localPosition = pos;
            if (gotRot) transform.localRotation = rot;

            if (Time.unscaledTime - lastDiagT > 3f)
            {
                Debug.Log($"[FARM CAM] src={src} gotPos={gotPos} pos={pos} gotRot={gotRot} world={transform.position}");
                lastDiagT = Time.unscaledTime;
            }
        }
    }
}
