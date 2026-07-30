import type { BodyChannel, CommandState, WorldSnapshot } from "./types";

export function liveCommands(snapshot: WorldSnapshot): CommandState[] {
  if (snapshot.active_commands && snapshot.active_commands.length > 0) {
    return snapshot.active_commands;
  }
  return snapshot.active_command ? [snapshot.active_command] : [];
}

export function focusedCommand(snapshot: WorldSnapshot): CommandState | null {
  return snapshot.active_command ?? liveCommands(snapshot).at(-1) ?? snapshot.last_command;
}

export function liveChannels(snapshot: WorldSnapshot): BodyChannel[] {
  return [...new Set(liveCommands(snapshot).flatMap((command) => command.channels))];
}
