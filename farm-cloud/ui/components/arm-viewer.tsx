"use client";

/**
 * <ArmViewer/> — react-three-fiber + urdf-loader.
 *
 * Loads /urdf/uf850/uf850.urdf (served from public/), keeps a single THREE
 * mesh tree alive across rerenders, and applies the latest joint state from
 * an SSE feed every frame. Scene props (table, blocks, cup) come from
 * /v1/scene; their live positions come from world snapshots.
 */

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Environment, Grid, ContactShadows } from "@react-three/drei";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import URDFLoader, { type URDFRobot } from "urdf-loader";
import {
  fetchScene,
  type SceneProp,
  type WorldSnapshot,
} from "@/lib/api";
import { subscribeSSE } from "@/lib/sse";

const URDF_PATH = "/urdf/uf850/uf850.urdf";
const ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"];
const FINGER_JOINT_NAMES = ["finger_left_joint", "finger_right_joint"];

type LiveState = {
  arm?: number[];
  fingers?: number[];
  worldProps?: WorldSnapshot["props"];
};

function ArmRobot({ live }: { live: React.MutableRefObject<LiveState> }) {
  const { scene } = useThree();
  const robotRef = useRef<URDFRobot | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const loader = new URDFLoader();
    // urdf-loader resolves mesh paths relative to the URDF; tell it where.
    loader.workingPath = "/urdf/uf850/";
    loader.parseCollision = false;
    loader.parseVisual = true;
    loader.load(URDF_PATH, (robot: URDFRobot) => {
      robot.rotation.x = -Math.PI / 2; // MuJoCo Z-up → three.js Y-up
      robot.scale.setScalar(1);
      scene.add(robot);
      robotRef.current = robot;
      setLoaded(true);
    });
    return () => {
      const r = robotRef.current;
      if (r) {
        scene.remove(r);
        r.traverse((o) => {
          if ((o as THREE.Mesh).geometry)
            (o as THREE.Mesh).geometry.dispose?.();
        });
      }
    };
  }, [scene]);

  useFrame(() => {
    const robot = robotRef.current;
    const arm = live.current.arm;
    if (!robot || !arm) return;
    for (let i = 0; i < ARM_JOINT_NAMES.length; i++) {
      const name = ARM_JOINT_NAMES[i];
      const value = arm[i];
      if (name == null || typeof value !== "number") continue;
      const j = robot.joints[name];
      if (!j) continue;
      (j as unknown as { setJointValue(v: number): void }).setJointValue(value);
    }
    const f = live.current.fingers;
    if (f) {
      for (let i = 0; i < FINGER_JOINT_NAMES.length; i++) {
        const name = FINGER_JOINT_NAMES[i];
        const value = f[i];
        if (name == null || typeof value !== "number") continue;
        const j = robot.joints[name];
        if (!j) continue;
        (j as unknown as { setJointValue(v: number): void }).setJointValue(value);
      }
    }
  });

  return loaded ? null : (
    <mesh position={[0, 0.6, 0]}>
      <boxGeometry args={[0.1, 0.1, 0.1]} />
      <meshStandardMaterial color="#888" />
    </mesh>
  );
}

function SceneProps({
  staticProps,
  live,
}: {
  staticProps: SceneProp[];
  live: React.MutableRefObject<LiveState>;
}) {
  // Render each prop as its own group whose pos+quat are written every frame
  // from the live world snapshot. Falls back to the static (initial) position
  // until the first snapshot lands.
  const groupRefs = useRef<Record<string, THREE.Group | null>>({});

  useFrame(() => {
    const worldProps = live.current.worldProps;
    if (!worldProps) return;
    for (const [id, ref] of Object.entries(groupRefs.current)) {
      const entry = worldProps[id];
      if (ref && entry) {
        // MuJoCo Z-up → three.js Y-up. The whole world is rotated -π/2 about X
        // at the viewer root so prop positions in MuJoCo coords map directly.
        ref.position.set(entry.pos[0], entry.pos[1], entry.pos[2]);
        const [w, x, y, z] = entry.quat as [number, number, number, number];
        ref.quaternion.set(x, y, z, w);
      }
    }
  });

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      {/* Table (matches the static fixture in the MJCF: top at z=0.26 m,
          centered at y=-0.70 m). */}
      <mesh position={[0, -0.70, 0.265]} receiveShadow>
        <boxGeometry args={[0.55, 0.45, 0.01]} />
        <meshStandardMaterial color="#c6a37a" />
      </mesh>
      {/* Legs */}
      {(
        [
          [0.25, -0.90, 0.065],
          [-0.25, -0.90, 0.065],
          [0.25, -0.50, 0.065],
          [-0.25, -0.50, 0.065],
        ] as [number, number, number][]
      ).map((p, i) => (
        <mesh key={i} position={p} receiveShadow>
          <boxGeometry args={[0.024, 0.024, 0.130]} />
          <meshStandardMaterial color="#695639" />
        </mesh>
      ))}
      {/* Props */}
      {staticProps.map((p) => {
        const rgba = p.rgba ?? [0.8, 0.8, 0.8, 1];
        const color = new THREE.Color(rgba[0], rgba[1], rgba[2]);
        const opacity = rgba[3] ?? 1;
        return (
          <group
            key={p.id}
            ref={(el) => {
              groupRefs.current[p.id] = el;
            }}
            position={p.pos}
          >
            {p.shape === "box" ? (
              <mesh castShadow receiveShadow>
                <boxGeometry
                  args={[
                    (p.size[0] ?? 0.0125) * 2,
                    (p.size[1] ?? 0.0125) * 2,
                    (p.size[2] ?? 0.0125) * 2,
                  ]}
                />
                <meshStandardMaterial
                  color={color}
                  transparent={opacity < 1}
                  opacity={opacity}
                />
              </mesh>
            ) : p.shape === "cylinder" ? (
              <mesh castShadow receiveShadow>
                <cylinderGeometry
                  args={[
                    p.size[0] ?? 0.04,
                    p.size[0] ?? 0.04,
                    (p.size[1] ?? 0.04) * 2,
                    32,
                  ]}
                />
                <meshStandardMaterial
                  color={color}
                  transparent={opacity < 1}
                  opacity={opacity}
                />
              </mesh>
            ) : null}
          </group>
        );
      })}
    </group>
  );
}

export function ArmViewer({
  worldStreamPath = "/v1/world/stream",
  height = 480,
}: {
  worldStreamPath?: string;
  height?: number;
}) {
  const liveRef = useRef<LiveState>({});
  const [staticProps, setStaticProps] = useState<SceneProp[]>([]);

  useEffect(() => {
    fetchScene().then((s) => {
      if (s) setStaticProps(s.props);
    });
  }, []);

  useEffect(() => {
    const es = subscribeSSE(worldStreamPath, (raw) => {
      const ev = raw as
        | { type: "joint_state"; arm?: number[]; fingers?: number[] }
        | ({ type: "world_snapshot" } & WorldSnapshot);
      if (!ev || typeof ev.type !== "string") return;
      if (ev.type === "joint_state") {
        liveRef.current = {
          ...liveRef.current,
          arm: ev.arm,
          fingers: ev.fingers,
        };
      } else if (ev.type === "world_snapshot") {
        liveRef.current = {
          arm: ev.joints,
          fingers: liveRef.current.fingers,
          worldProps: ev.props,
        };
      }
    });
    return () => es.close();
  }, [worldStreamPath]);

  return (
    <div
      style={{
        width: "100%",
        height,
        borderRadius: 12,
        overflow: "hidden",
        background: "linear-gradient(180deg, #f7f6f3 0%, #e3e0d8 100%)",
        boxShadow: "0 1px 0 rgba(0,0,0,0.05), 0 8px 24px rgba(0,0,0,0.08)",
      }}
    >
      <Canvas shadows camera={{ position: [1.5, 1.2, 1.5], fov: 38 }}>
        <Suspense fallback={null}>
          <ambientLight intensity={0.45} />
          <directionalLight
            position={[2, 3, 1]}
            intensity={1.1}
            castShadow
            shadow-mapSize-width={1024}
            shadow-mapSize-height={1024}
            shadow-camera-far={6}
            shadow-camera-left={-2}
            shadow-camera-right={2}
            shadow-camera-top={2}
            shadow-camera-bottom={-2}
          />
          <Environment preset="apartment" />
          <Grid
            position={[0, 0.001, 0]}
            args={[6, 6]}
            cellSize={0.1}
            cellThickness={0.5}
            cellColor="#bcbcbc"
            sectionSize={0.5}
            sectionThickness={1}
            sectionColor="#7a7a7a"
            fadeDistance={4}
            fadeStrength={1}
            infiniteGrid
          />
          <ContactShadows
            position={[0, 0.002, 0]}
            opacity={0.35}
            scale={4}
            blur={2}
            far={1}
          />
          <ArmRobot live={liveRef} />
          <SceneProps staticProps={staticProps} live={liveRef} />
          <OrbitControls
            makeDefault
            target={[0, 0.30, 0.55]}
            enableDamping
            dampingFactor={0.08}
            minDistance={0.6}
            maxDistance={4}
          />
        </Suspense>
      </Canvas>
    </div>
  );
}

export default ArmViewer;
