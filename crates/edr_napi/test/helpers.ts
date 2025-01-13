import { toBytes } from "@nomicfoundation/ethereumjs-util";
import {
  EdrContext,
  TracingMessage,
  TracingMessageResult,
  TracingStep,
} from "..";

function getEnv(key: string): string | undefined {
  const variable = process.env[key];
  if (variable === undefined || variable === "") {
    return undefined;
  }

  const trimmed = variable.trim();

  return trimmed.length === 0 ? undefined : trimmed;
}

export const ALCHEMY_URL = getEnv("ALCHEMY_URL");

export function isCI(): boolean {
  return getEnv("CI") === "true";
}

let context: EdrContext | undefined;

export function getContext(): EdrContext {
  if (context === undefined) {
    context = new EdrContext();
  }
  return context;
}

/**
 * Given a trace, return only its steps.
 */
export function collectSteps(
  trace: Array<TracingMessage | TracingStep | TracingMessageResult>
): TracingStep[] {
  return trace.filter((traceItem) => "pc" in traceItem) as TracingStep[];
}

/**
 * Given a trace, return only its messages.
 */
export function collectMessages(
  trace: Array<TracingMessage | TracingStep | TracingMessageResult>
): TracingMessage[] {
  return trace.filter(
    (traceItem) => "isStaticCall" in traceItem
  ) as TracingMessage[];
}

export function toBuffer(x: Parameters<typeof toBytes>[0]) {
  return Buffer.from(toBytes(x));
}
