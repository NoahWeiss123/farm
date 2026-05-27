// Left-hand UI ray pointer for the FARM floating tablet.
//
// The right-hand instance stays a no-op (passthrough already shows the
// user's actual right hand on the controller, which drives the arm).
// The left-hand instance casts a line at the HUD card every frame,
// hit-tests it against the card's button rects, and fires the button's
// onClick when the left trigger is pulled.

using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

namespace TeleopDataCollector
{
    public class ControllerPointer : MonoBehaviour
    {
        public bool IsRight = true;

        LineRenderer line;
        Transform dot;
        Renderer dotRenderer;
        bool prevTrigger;

        static readonly Color C_LINE       = new Color(0.55f, 0.78f, 0.96f, 0.85f);
        static readonly Color C_DOT_IDLE   = new Color(0.55f, 0.78f, 0.96f, 1f);
        static readonly Color C_DOT_HOVER  = new Color(0.40f, 0.95f, 0.60f, 1f);
        const float RAY_MAX_M = 5f;

        void Awake()
        {
            if (IsRight) { enabled = false; return; }
            BuildVisuals();
        }

        void BuildVisuals()
        {
            line = gameObject.AddComponent<LineRenderer>();
            line.useWorldSpace = true;
            line.positionCount = 2;
            line.startWidth = 0.004f;
            line.endWidth = 0.002f;
            line.material = new Material(Shader.Find("Sprites/Default"));
            line.startColor = C_LINE;
            line.endColor = C_LINE;
            line.numCapVertices = 2;
            line.enabled = false;

            var dotGO = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            dotGO.name = "ui_pointer_dot";
            dotGO.transform.SetParent(transform, false);
            dotGO.transform.localScale = Vector3.one * 0.012f;
            var collider = dotGO.GetComponent<Collider>();
            if (collider != null) Destroy(collider);
            dot = dotGO.transform;
            dotRenderer = dotGO.GetComponent<Renderer>();
            dotRenderer.material = new Material(Shader.Find("Sprites/Default"));
            dotRenderer.material.color = C_DOT_IDLE;
            dot.gameObject.SetActive(false);
        }

        void Update()
        {
            var hud = Object.FindFirstObjectByType<HUDController>();
            if (hud == null || hud.CardTransform == null)
            {
                if (line != null) line.enabled = false;
                if (dot != null) dot.gameObject.SetActive(false);
                return;
            }

            var left = InputDevices.GetDeviceAtXRNode(XRNode.LeftHand);
            if (!left.isValid)
            {
                var ll = new List<InputDevice>();
                InputDevices.GetDevicesWithCharacteristics(
                    InputDeviceCharacteristics.Left | InputDeviceCharacteristics.Controller, ll);
                if (ll.Count > 0) left = ll[0];
            }
            if (!left.isValid ||
                !left.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 lpos) ||
                !left.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion lrot))
            {
                line.enabled = false;
                dot.gameObject.SetActive(false);
                return;
            }

            // Controller-space origin/forward sit in the tracking-rig frame.
            // Convert into world by applying the rig's transform — that's
            // the same parent the HUD card uses, so the math agrees with
            // the card's world pose.
            var rig = Camera.main != null ? Camera.main.transform.parent : null;
            Vector3 originW = rig != null ? rig.TransformPoint(lpos) : lpos;
            Quaternion rotW = rig != null ? rig.rotation * lrot : lrot;
            Vector3 dirW = rotW * Vector3.forward;

            // Ray-plane intersection with the card plane. The canvas's
            // local +Z (transform.forward) points AWAY from the reader, so
            // the plane normal pointing toward the user is -forward.
            var card = hud.CardTransform;
            Vector3 planePos = card.position;
            Vector3 planeNormal = -card.forward;
            float denom = Vector3.Dot(dirW, planeNormal);
            bool hit = false;
            Vector3 hitW = Vector3.zero;
            Vector2 localXY = Vector2.zero;
            if (Mathf.Abs(denom) > 1e-5f)
            {
                float t = Vector3.Dot(planePos - originW, planeNormal) / denom;
                if (t > 0f && t < RAY_MAX_M)
                {
                    hitW = originW + dirW * t;
                    // Convert to canvas-local 2D (in canvas pixels).
                    Vector3 localHit = card.InverseTransformPoint(hitW);
                    localXY = new Vector2(localHit.x, localHit.y);
                    Vector2 half = hud.CardSizePx * 0.5f;
                    if (Mathf.Abs(localXY.x) <= half.x && Mathf.Abs(localXY.y) <= half.y)
                        hit = true;
                }
            }

            line.enabled = true;
            line.SetPosition(0, originW);
            line.SetPosition(1, hit ? hitW : originW + dirW * 0.6f);
            dot.gameObject.SetActive(hit);
            if (hit) dot.position = hitW;

            // Hit-test against the card's button rects.
            HUDController.UIButton? hovered = null;
            if (hit)
            {
                foreach (var b in hud.Buttons)
                {
                    if (b.rect == null) continue;
                    Vector2 c = b.rect.anchoredPosition;
                    Vector2 s = b.rect.sizeDelta * 0.5f;
                    if (localXY.x >= c.x - s.x && localXY.x <= c.x + s.x &&
                        localXY.y >= c.y - s.y && localXY.y <= c.y + s.y)
                    {
                        hovered = b;
                        break;
                    }
                }
            }
            if (dotRenderer != null)
                dotRenderer.material.color = hovered.HasValue ? C_DOT_HOVER : C_DOT_IDLE;

            // Rising-edge click on left trigger.
            left.TryGetFeatureValue(CommonUsages.trigger, out float trig);
            bool triggerDown = trig > 0.65f;
            if (triggerDown && !prevTrigger && hovered.HasValue)
                hovered.Value.onClick?.Invoke();
            prevTrigger = triggerDown;
        }
    }
}
