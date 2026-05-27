// Direct TCP publisher to FARM's bridge — bypasses ROSConnection so we
// don't have to implement its handshake protocol server-side. Speaks the
// 4-byte-topic-len + topic + 4-byte-body-len + body frame format defined
// in farm_edge_agent/ros_bridge/wire.py, with message layouts matching
// farm_edge_agent/ros_bridge/messages.py (PoseStamped, OVR2ROSInputs).

using System;
using System.IO;
using System.Net.Sockets;
using UnityEngine;
using UnityEngine.XR;

namespace TeleopDataCollector
{
    public class Q2RPublisher : MonoBehaviour
    {
        const string TOPIC_POSE   = "/q2r_right_hand_pose";
        const string TOPIC_INPUTS = "/q2r_right_hand_inputs";

        [Range(20f, 200f)] public float publishHzCap = 200f;

        public bool IsLinked => stream != null;

        TcpClient client;
        NetworkStream stream;
        float retryAt;
        float lastSendT;
        float lastDiagT;
        int sentCount;
        InputDevice right;

        // Edge-detect button presses so the HUD can mirror the bridge's
        // recording state without round-tripping to the laptop. Both
        // sides apply the same A/B logic to the same edges, so they
        // can't drift out of sync.
        bool prevA, prevB;
        float recordStartTime;

        void OnEnable()
        {
            ConnectionConfig.LoadFromPrefs();
            TryConnect();
        }

        void OnDisable()
        {
            CloseLink();
        }

        void TryConnect()
        {
            try
            {
                CloseLink();
                client = new TcpClient { NoDelay = true };
                var ar = client.BeginConnect(ConnectionConfig.Ip, ConnectionConfig.Port, null, null);
                if (!ar.AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(1.5)))
                    throw new TimeoutException("connect timed out");
                client.EndConnect(ar);
                stream = client.GetStream();
                Debug.Log($"[FARM] tcp link up {ConnectionConfig.Ip}:{ConnectionConfig.Port}");
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[FARM] connect failed: {e.Message}");
                CloseLink();
                retryAt = Time.unscaledTime + 2f;
            }
        }

        void CloseLink()
        {
            try { stream?.Close(); } catch {}
            try { client?.Close(); } catch {}
            stream = null; client = null;
        }

        void Update()
        {
            float now = Time.unscaledTime;

            // Re-fetch every frame; InputDevice is a struct and a stale
            // copy can keep isValid=false forever.
            right = InputDevices.GetDeviceAtXRNode(XRNode.RightHand);
            if (!right.isValid)
            {
                var rl = new System.Collections.Generic.List<InputDevice>();
                InputDevices.GetDevicesWithCharacteristics(
                    InputDeviceCharacteristics.Right | InputDeviceCharacteristics.Controller, rl);
                if (rl.Count > 0) right = rl[0];
            }

            if (now - lastDiagT > 2f)
            {
                var all = new System.Collections.Generic.List<InputDevice>();
                InputDevices.GetDevices(all);
                Debug.Log($"[FARM] hb linked={stream != null} sent={sentCount} right_valid={right.isValid} all_devs={all.Count}");
                foreach (var d in all)
                    Debug.Log($"[FARM]   dev name='{d.name}' mfg='{d.manufacturer}' char={d.characteristics} valid={d.isValid}");
                lastDiagT = now;
            }

            if (stream == null)
            {
                if (now >= retryAt) TryConnect();
                return;
            }

            float minDt = 1f / Mathf.Max(1f, publishHzCap);
            if (now - lastSendT < minDt) return;
            lastSendT = now;

            if (!right.isValid) return;
            if (!right.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 pos)) return;
            if (!right.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion rot)) return;
            right.TryGetFeatureValue(CommonUsages.trigger, out float trigger);
            right.TryGetFeatureValue(CommonUsages.grip,    out float grip);
            // A = primary (lower), B = secondary (upper) — Touch Plus.
            right.TryGetFeatureValue(CommonUsages.primaryButton,   out bool buttonLower);
            right.TryGetFeatureValue(CommonUsages.secondaryButton, out bool buttonUpper);
            right.TryGetFeatureValue(CommonUsages.primary2DAxisClick, out bool stickClick);

            // Unity (left-handed, Y-up) → ROS REP-103 FLU.
            double rx = pos.z,  ry = -pos.x, rz = pos.y;
            double qx = -rot.z, qy = rot.x, qz = -rot.y, qw = rot.w;

            // Mirror the bridge's recording state machine so the in-headset
            // HUD timer matches what the laptop is actually doing.
            if (buttonLower && !prevA)
            {
                if (HUDController.RecordingActive)
                {
                    HUDController.RecordingActive = false;
                    HUDController.RecordingElapsedS = 0f;
                }
                else
                {
                    HUDController.RecordingActive = true;
                    recordStartTime = Time.unscaledTime;
                }
            }
            if (buttonUpper && !prevB && HUDController.RecordingActive)
            {
                HUDController.RecordingActive = false;
                HUDController.RecordingElapsedS = 0f;
            }
            prevA = buttonLower;
            prevB = buttonUpper;
            if (HUDController.RecordingActive)
                HUDController.RecordingElapsedS = Time.unscaledTime - recordStartTime;

            try
            {
                SendPoseStamped(rx, ry, rz, qx, qy, qz, qw);
                SendInputs(trigger, grip, buttonLower, buttonUpper, stickClick);
                sentCount++;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[FARM] send failed: {e.Message}");
                CloseLink();
                retryAt = Time.unscaledTime + 1f;
            }
        }

        // ─── wire helpers ───────────────────────────────────────────────

        void WriteFrame(string topic, byte[] body)
        {
            byte[] topicBytes = System.Text.Encoding.UTF8.GetBytes(topic);
            byte[] tlen = BitConverter.GetBytes((uint)topicBytes.Length);
            byte[] blen = BitConverter.GetBytes((uint)body.Length);
            stream.Write(tlen, 0, 4);
            stream.Write(topicBytes, 0, topicBytes.Length);
            stream.Write(blen, 0, 4);
            stream.Write(body, 0, body.Length);
        }

        void SendPoseStamped(double px, double py, double pz,
                             double qx, double qy, double qz, double qw)
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            bw.Write((int)0); bw.Write((uint)0); bw.Write((uint)0);
            bw.Write(px); bw.Write(py); bw.Write(pz);
            bw.Write(qx); bw.Write(qy); bw.Write(qz); bw.Write(qw);
            WriteFrame(TOPIC_POSE, ms.ToArray());
        }

        void SendInputs(float trigger, float grip, bool buttonLower, bool buttonUpper, bool stickClick)
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            // OVR2ROSInputs wire order: button_upper(b), button_lower(b),
            // stick.x(f32), stick.y(f32), press_index(f32), press_middle(f32),
            // thumb_stick_click(b).
            bw.Write((byte)(buttonUpper ? 1 : 0));
            bw.Write((byte)(buttonLower ? 1 : 0));
            bw.Write((float)0);   // thumb stick H — unused
            bw.Write((float)0);   // thumb stick V — unused
            bw.Write(trigger);    // press_index
            bw.Write(grip);       // press_middle
            bw.Write((byte)(stickClick ? 1 : 0));
            WriteFrame(TOPIC_INPUTS, ms.ToArray());
        }
    }
}
