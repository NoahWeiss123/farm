import { Hono } from "hono";
import type { Env } from "./env";

export function registerRoutes(app: Hono<{ Bindings: Env }>): void {
  app.post("/v1/plans", (c) =>
    c.json({ error: "not_implemented", route: "planner" }, 501),
  );

  app.post("/v1/runs/:id/dispatch", (c) =>
    c.json(
      { error: "not_implemented", route: "dispatcher", run_id: c.req.param("id") },
      501,
    ),
  );

  app.get("/v1/runs/:id", (c) =>
    c.json(
      { error: "not_implemented", route: "session", run_id: c.req.param("id") },
      501,
    ),
  );
}
