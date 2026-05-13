import { describe, expect, it, vi, afterEach } from "vitest";
import { act, cleanup, render, screen } from "@testing-library/react";
import { Nav } from "@/components/nav";
import RunsPage from "@/app/runs/page";
import { WarmingUp } from "@/app/runs/[id]/page";

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

describe("RunsPage empty state", () => {
  it("renders the run-your-first-task card when there are no runs", async () => {
    const page = await RunsPage();
    render(page);
    expect(screen.getByText("Run your first task")).not.toBeNull();
    const cta = screen.getByRole("link", {
      name: "Read the getting-started guide",
    });
    expect(cta.getAttribute("href")).toBe("/docs/getting-started");
  });
});

describe("WarmingUp", () => {
  it("shows the warming-up copy and an elapsed counter that ticks", () => {
    vi.useFakeTimers();
    try {
      render(<WarmingUp runId="r-123" />);
      expect(screen.getByText("Warming up...")).not.toBeNull();
      expect(screen.getByText("Run r-123")).not.toBeNull();
      expect(screen.getByTestId("elapsed").textContent).toBe("Elapsed: 0s");

      act(() => {
        vi.advanceTimersByTime(1000);
      });
      expect(screen.getByTestId("elapsed").textContent).toBe("Elapsed: 1s");

      act(() => {
        vi.advanceTimersByTime(2000);
      });
      expect(screen.getByTestId("elapsed").textContent).toBe("Elapsed: 3s");
    } finally {
      vi.useRealTimers();
    }
  });
});
