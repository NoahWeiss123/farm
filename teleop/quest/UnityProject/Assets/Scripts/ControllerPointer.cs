// Left-hand UI navigator for the FARM floating tablet.
//
// The right-hand instance is a no-op (right hand drives the arm).
// The left-hand instance:
//   • Tilt left stick horizontally → move selection between HUD buttons
//   • Pull the left trigger → invoke selected button
//
// No laser. The HUD draws a highlight ring around the selected button.

using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

namespace TeleopDataCollector
{
    public class ControllerPointer : MonoBehaviour
    {
        public bool IsRight = true;

        const float STICK_DEADZONE = 0.5f;
        const float STICK_REPEAT_S = 0.25f;
        const float TRIGGER_DOWN   = 0.6f;

        HUDController hud;
        float lastStickMoveT;
        bool prevTrigger;
        float lastDiagT;

        void Awake()
        {
            if (IsRight) { enabled = false; return; }
        }

        void Update()
        {
            if (hud == null)
            {
                hud = Object.FindFirstObjectByType<HUDController>();
                if (hud == null) return;
            }

            var left = InputDevices.GetDeviceAtXRNode(XRNode.LeftHand);
            if (!left.isValid)
            {
                var ll = new List<InputDevice>();
                InputDevices.GetDevicesWithCharacteristics(
                    InputDeviceCharacteristics.Left | InputDeviceCharacteristics.Controller, ll);
                if (ll.Count > 0) left = ll[0];
            }
            if (!left.isValid) return;

            // Stick navigation — horizontal axis moves selection. Re-arm only
            // when the stick returns to neutral so a held tilt doesn't walk
            // selection forever; STICK_REPEAT_S still steps periodically.
            // primary2DAxis is the canonical Quest thumbstick; some bindings
            // expose it as secondary2DAxis, so fall back if the primary is
            // missing.
            bool gotStick = left.TryGetFeatureValue(CommonUsages.primary2DAxis, out Vector2 stick);
            if (!gotStick || stick == Vector2.zero)
                left.TryGetFeatureValue(CommonUsages.secondary2DAxis, out stick);

            float now = Time.unscaledTime;
            if (Mathf.Abs(stick.x) > STICK_DEADZONE && now - lastStickMoveT > STICK_REPEAT_S)
            {
                int step = stick.x > 0 ? +1 : -1;
                int next = Mathf.Clamp(hud.SelectedIndex + step, 0, hud.Buttons.Count - 1);
                if (next != hud.SelectedIndex)
                {
                    hud.SelectedIndex = next;
                    lastStickMoveT = now;
                }
            }
            else if (Mathf.Abs(stick.x) <= STICK_DEADZONE)
            {
                lastStickMoveT = 0f;
            }

            // Left trigger click. Read both the analog axis and the boolean
            // button bind — whichever the runtime exposes first.
            bool gotTrig = left.TryGetFeatureValue(CommonUsages.trigger, out float trig);
            left.TryGetFeatureValue(CommonUsages.triggerButton, out bool trigBtn);
            bool triggerDown = trigBtn || (gotTrig && trig >= TRIGGER_DOWN);
            if (triggerDown && !prevTrigger) hud.InvokeSelected();
            prevTrigger = triggerDown;

            if (now - lastDiagT > 2f)
            {
                Debug.Log($"[FARM PTR] stick={stick} (gotStick={gotStick}) trig={trig:F2} trigBtn={trigBtn} sel={hud.SelectedIndex}");
                lastDiagT = now;
            }
        }
    }
}
