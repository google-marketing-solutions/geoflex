<!--
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->
<template>
  <div>
    <q-separator height="3" />
    <q-list separator>
      <q-item clickable v-ripple to="/configuration">
        <q-item-section>
          <q-item-label>Configuration</q-item-label>
          <q-item-label caption lines="2">Manage configuration</q-item-label>
        </q-item-section>
        <q-item-section side>
          <q-icon name="settings" />
        </q-item-section>
      </q-item>

      <q-item clickable v-ripple to="/datasources">
        <q-item-section>
          <q-item-label>Data Sources</q-item-label>
          <q-item-label caption lines="2">Define and manage data sources</q-item-label>
        </q-item-section>
        <q-item-section side>
          <q-icon name="source" />
        </q-item-section>
      </q-item>
    </q-list>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue';

export default defineComponent({
  name: 'SideMenu',
});
</script>
