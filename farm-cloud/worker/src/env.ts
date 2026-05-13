export interface Env {
  // Durable Object namespaces — wired by wrangler.toml in task 017.
  // DISPATCHER: DurableObjectNamespace;
  // SESSION: DurableObjectNamespace;
}

export const WORKER_VERSION = "0.0.1";
export const PROTOCOL_VERSION = "1.2.0";
