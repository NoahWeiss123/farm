import { Hono } from "hono";
import { PROTOCOL_VERSION, WORKER_VERSION, type Env } from "./env";
import { registerRoutes } from "./routes";

const app = new Hono<{ Bindings: Env }>();

app.get("/", (c) => c.text("farm-planner-worker v0"));

app.get("/healthz", (c) =>
  c.json({
    ok: true,
    version: WORKER_VERSION,
    protocol_version: PROTOCOL_VERSION,
  }),
);

registerRoutes(app);

export default app;
