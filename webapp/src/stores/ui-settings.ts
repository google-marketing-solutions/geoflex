/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { defineStore } from 'pinia';
import { ref, watch } from 'vue';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type ComponentSettings = Record<string, any>;

export const useUiSettingsStore = defineStore('uiSettings', () => {
  const settings = ref<Record<string, ComponentSettings>>({});

  // Load settings from localStorage
  function loadSettings() {
    const storedSettings = localStorage.getItem('uiSettings');
    if (storedSettings) {
      settings.value = JSON.parse(storedSettings);
    }
  }

  // Watch for changes and save to localStorage
  watch(
    settings,
    (newSettings) => {
      localStorage.setItem('uiSettings', JSON.stringify(newSettings));
    },
    { deep: true },
  );

  function getComponentSettings(componentId: string): ComponentSettings {
    return settings.value[componentId] || {};
  }

  function saveComponentSettings(componentId: string, newSettings: ComponentSettings) {
    settings.value[componentId] = { ...settings.value[componentId], ...newSettings };
  }

  // Initial load
  loadSettings();

  return {
    settings,
    getComponentSettings,
    saveComponentSettings,
  };
});
