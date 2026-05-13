import { Hono } from "hono";
import type { Env } from "./env";
import { Dispatcher } from "./dispatcher";

export { Dispatcher };

const app = new Hono<{ Bindings: Env }>();

app.get("/v1/runs/:id", async (c) => {
  const id = c.req.param("id");
  const stub = c.env.DISPATCHER.get(c.env.DISPATCHER.idFromName(id));
  const url = new URL(c.req.url);
  return stub.fetch(`${url.origin}/runs/${id}`);
});

app.post("/v1/runs/:id/dispatch", async (c) => {
  const id = c.req.param("id");
  const stub = c.env.DISPATCHER.get(c.env.DISPATCHER.idFromName(id));
  const url = new URL(c.req.url);
  return stub.fetch(`${url.origin}/run`, {
    method: "POST",
    body: await c.req.text(),
    headers: { "content-type": "application/json" },
  });
});

export default app;
