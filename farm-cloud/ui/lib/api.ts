export interface RunSummary {
  id: string;
  status: string;
}

export interface RunDetail {
  id: string;
  status: string;
}

export async function fetchRun(_id: string): Promise<RunDetail | null> {
  return null;
}

export async function fetchRuns(): Promise<RunSummary[] | null> {
  return null;
}
