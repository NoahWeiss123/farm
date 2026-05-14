import { describe, expect, it, afterEach, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { Nav } from "@/components/nav";

afterEach(() => {
  cleanup();
});

describe("Nav", () => {
  it("renders Runs and Docs links", () => {
    render(<Nav />);
    const runs = screen.getByRole("link", { name: "Runs" });
    const docs = screen.getByRole("link", { name: "Docs" });
    expect(runs.getAttribute("href")).toBe("/runs");
    expect(docs.getAttribute("href")).toBe("/docs");
  });
});

describe("lib/api shape", () => {
  it("exposes the expected client functions", async () => {
    const mod = await import("@/lib/api");
    expect(typeof mod.fetchRuns).toBe("function");
    expect(typeof mod.fetchRun).toBe("function");
    expect(typeof mod.submitRun).toBe("function");
    expect(typeof mod.fetchScene).toBe("function");
    expect(typeof mod.fetchWorld).toBe("function");
  });

  it("submitRun returns null and does not throw when the daemon is unreachable", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("ECONNREFUSED")) as typeof fetch;
    try {
      const mod = await import("@/lib/api");
      const out = await mod.submitRun("any task");
      expect(out).toBe(null);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
