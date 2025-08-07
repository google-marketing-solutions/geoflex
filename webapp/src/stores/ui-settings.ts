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
