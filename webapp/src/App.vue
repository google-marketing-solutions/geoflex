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
  <router-view />
</template>

<script setup lang="ts">
import { useQuasar } from 'quasar';
import { onMounted, watch } from 'vue';

const $q = useQuasar();

defineOptions({
  name: 'App',
});

// Load preference on component mount
onMounted(() => {
  const darkModePreference = localStorage.getItem('darkMode');
  if (darkModePreference !== null) {
    $q.dark.set(darkModePreference === 'true');
  }
});

// Save preference whenever it changes
watch(
  () => $q.dark.isActive,
  (isDark) => {
    localStorage.setItem('darkMode', isDark.toString());
  },
);
</script>
