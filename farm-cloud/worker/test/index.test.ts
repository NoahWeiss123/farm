import { describe, expect, it } from "vitest";
import app from "../src/index";
import { PROTOCOL_VERSION, WORKER_VERSION } from "../src/env";

describe("root", () => {
  it("GET / returns the hello string", async () => {
    const res = await app.request("/");
    expect(res.status).toBe(200);
    expect(await res.text()).toBe("farm-planner-worker v0");
  });
});

describe("healthz", () => {
  it("returns 200 with ok: true and version fields", async () => {
    const res = await app.request("/healthz");
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      ok: boolean;
      version: string;
      protocol_version: string;
    };
    expect(body.ok).toBe(true);
    expect(body.version).toBe(WORKER_VERSION);
    expect(body.protocol_version).toBe(PROTOCOL_VERSION);
  });
});

describe("stub routes", () => {
  it("POST /v1/plans returns 501", async () => {
    const res = await app.request("/v1/plans", { method: "POST" });
    expect(res.status).toBe(501);
    const body = (await res.json()) as { error: string; route: string };
    expect(body.error).toBe("not_implemented");
    expect(body.route).toBe("planner");
  });

  it("POST /v1/runs/:id/dispatch returns 501 and echoes run_id", async () => {
    const res = await app.request("/v1/runs/r_abc/dispatch", { method: "POST" });
    expect(res.status).toBe(501);
    const body = (await res.json()) as {
      error: string;
      route: string;
      run_id: string;
    };
    expect(body.error).toBe("not_implemented");
    expect(body.route).toBe("dispatcher");
    expect(body.run_id).toBe("r_abc");
  });

  it("GET /v1/runs/:id returns 501 and echoes run_id", async () => {
    const res = await app.request("/v1/runs/r_xyz");
    expect(res.status).toBe(501);
    const body = (await res.json()) as {
      error: string;
      route: string;
      run_id: string;
    };
    expect(body.error).toBe("not_implemented");
    expect(body.route).toBe("session");
    expect(body.run_id).toBe("r_xyz");
  });

  it("unknown route returns 404", async () => {
    const res = await app.request("/does-not-exist");
    expect(res.status).toBe(404);
  });
});
