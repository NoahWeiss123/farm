"use client";

import { useEffect, useRef, useState } from "react";
import { FARM_API } from "@/lib/api";

/**
 * <CameraTile/> — polls a daemon camera URL, swaps the bound <img> only
 * after the next frame has decoded so the UI never flashes a blank.
 *
 * The MuJoCo renderer is synchronous on the daemon's event loop; at 10 Hz
 * a 480×360 JPEG is ~12 KB, so the bandwidth is negligible.
 */
export function CameraTile({
  name,
  label,
  variant = "rgb",
  intervalMs = 100,
  aspect = "4 / 3",
}: {
  name: "exterior" | "wrist" | "topdown";
  label: string;
  variant?: "rgb" | "depth";
  intervalMs?: number;
  aspect?: string;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const tokRef = useRef(0);

  useEffect(() => {
    const ext = variant === "depth" ? "depth.png" : "jpg";
    let cancelled = false;
    const tick = () => {
      if (cancelled) return;
      const t = ++tokRef.current;
      const url = `${FARM_API}/v1/cameras/${name}.${ext}?t=${Date.now()}`;
      // Preload via Image() so we only swap when the bitmap is ready.
      const img = new Image();
      img.onload = () => {
        if (!cancelled && t === tokRef.current) setSrc(url);
      };
      img.onerror = () => {
        // Daemon offline / camera not yet wired — leave whatever's there.
      };
      img.src = url;
    };
    tick();
    const id = setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [name, variant, intervalMs]);

  return (
    <figure className="cam-tile" style={{ aspectRatio: aspect }}>
      {src ? (
        <img src={src} alt={`${label} camera`} />
      ) : (
        <div className="cam-skeleton" />
      )}
      <figcaption>
        <span className="cam-name">{label}</span>
        <span className="cam-meta">{variant === "depth" ? "depth" : "rgb"}</span>
      </figcaption>
    </figure>
  );
}

export default CameraTile;
