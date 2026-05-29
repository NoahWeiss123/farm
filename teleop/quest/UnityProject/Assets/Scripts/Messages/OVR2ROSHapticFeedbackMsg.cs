// Hand-written C# of quest2ros/OVR2ROSHapticFeedback. Subscribed only.

using System;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;

namespace RosMessageTypes.Quest2ros
{
    [Serializable]
    public class OVR2ROSHapticFeedbackMsg : Message
    {
        public const string k_RosMessageName = "quest2ros/OVR2ROSHapticFeedback";
        public override string RosMessageName => k_RosMessageName;

        public float frequency;
        public float amplitude;

        public OVR2ROSHapticFeedbackMsg() { }
        public OVR2ROSHapticFeedbackMsg(float frequency, float amplitude)
        {
            this.frequency = frequency;
            this.amplitude = amplitude;
        }

        public static OVR2ROSHapticFeedbackMsg Deserialize(MessageDeserializer d) =>
            new OVR2ROSHapticFeedbackMsg(d);

        OVR2ROSHapticFeedbackMsg(MessageDeserializer d)
        {
            d.Read(out this.frequency);
            d.Read(out this.amplitude);
        }

        public override void SerializeTo(MessageSerializer s)
        {
            s.Write(this.frequency);
            s.Write(this.amplitude);
        }

        public override string ToString() => $"OVR2ROSHapticFeedback(f={frequency:F1}Hz amp={amplitude:F2})";

#if UNITY_EDITOR
        [UnityEditor.InitializeOnLoadMethod]
#else
        [UnityEngine.RuntimeInitializeOnLoadMethod]
#endif
        public static void Register() => MessageRegistry.Register(k_RosMessageName, Deserialize);
    }
}
