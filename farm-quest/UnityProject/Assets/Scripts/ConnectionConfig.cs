// PlayerPrefs-backed config for ROS-TCP host. Boots ROSConnection with the saved
// IP/port and exposes setters used by the in-headset settings panel.

using UnityEngine;

namespace TeleopDataCollector
{
    public static class ConnectionConfig
    {
        const string KEY_IP   = "farm.quest.ip";
        const string KEY_PORT = "farm.quest.port";

        // Default host IP is this Mac's LAN address; in-headset picker
        // (left A+B) overrides at runtime. Port 10000 = FARM ROS-TCP-Endpoint
        // bridge (farm-edge-agent/src/farm_edge_agent/ros_bridge/).
        public const string DEFAULT_IP   = "10.32.82.173";
        public const int    DEFAULT_PORT = 10000;

        public static string Ip   { get; private set; } = DEFAULT_IP;
        public static int    Port { get; private set; } = DEFAULT_PORT;

        public static event System.Action OnChanged;

        public static void LoadFromPrefs()
        {
            Ip   = PlayerPrefs.GetString(KEY_IP,   DEFAULT_IP);
            Port = PlayerPrefs.GetInt   (KEY_PORT, DEFAULT_PORT);
        }

        public static void Apply(string ip, int port)
        {
            Ip = string.IsNullOrWhiteSpace(ip) ? DEFAULT_IP : ip.Trim();
            Port = port > 0 ? port : DEFAULT_PORT;
            PlayerPrefs.SetString(KEY_IP, Ip);
            PlayerPrefs.SetInt   (KEY_PORT, Port);
            PlayerPrefs.Save();
            OnChanged?.Invoke();
            Debug.Log($"[FARM] FARM endpoint set -> {Ip}:{Port}");
        }
    }
}
