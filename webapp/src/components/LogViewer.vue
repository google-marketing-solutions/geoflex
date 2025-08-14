<!--
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

<template>
  <q-card class="log-viewer-card" flat bordered>
    <q-card-section>
      <div class="text-h6">Logs</div>
    </q-card-section>

    <q-separator />

    <q-card-section class="q-pa-none">
      <q-list separator>
        <q-item
          v-for="(entry, index) in logs"
          :key="index"
          :class="`log-entry--${entry.level.toLowerCase()}`"
        >
          <q-item-section avatar>
            <q-icon
              :name="getIconForSeverity(entry.level)"
              :color="getColorForSeverity(entry.level)"
            />
          </q-item-section>
          <q-item-section>
            <q-item-label class="text-weight-bold"
              >[{{ formatTimestamp(entry.timestamp) }}]</q-item-label
            >
            <q-item-label>{{ entry.message }}</q-item-label>
          </q-item-section>
        </q-item>
      </q-list>
      <div v-if="!logs || logs.length === 0" class="q-pa-md text-center text-grey">
        No log entries to display.
      </div>
    </q-card-section>
  </q-card>
</template>

<script setup lang="ts">
import type { LogEntry } from 'src/components/models';

defineProps<{
  logs: LogEntry[];
}>();

const getIconForSeverity = (level: string) => {
  switch (level.toLowerCase()) {
    case 'info':
      return 'info';
    case 'warning':
      return 'warning';
    case 'error':
      return 'error';
    case 'debug':
      return 'bug_report';
    default:
      return 'help';
  }
};

const getColorForSeverity = (level: string) => {
  switch (level.toLowerCase()) {
    case 'info':
      return 'primary';
    case 'warning':
      return 'warning';
    case 'error':
      return 'negative';
    case 'debug':
      return 'grey';
    default:
      return 'grey';
  }
};

const formatTimestamp = (timestamp: string) => {
  return new Date(timestamp).toLocaleString();
};
</script>

<style lang="scss" scoped>
.log-viewer-card {
  max-height: 400px;
  display: flex;
  flex-direction: column;

  .q-card__section:last-child {
    flex-grow: 1;
    overflow-y: auto;
  }
}
</style>
