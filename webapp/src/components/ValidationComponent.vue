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
  <q-card class="q-mt-md q-pa-md">
    <div class="text-h6 q-mb-md">Data Source Validation</div>

    <!-- Metrics Validation -->
    <div class="q-mb-md">
      <div class="text-subtitle1 q-mb-sm">Metrics Mismatch</div>
      <q-banner v-if="validation.metrics.missing.length === 0" class="bg-positive text-white">
        All required metrics are present in the data source.
      </q-banner>
      <q-banner v-else class="bg-negative text-white">
        <ul>
          <li v-for="metric in validation.metrics.missing" :key="metric">{{ metric }}</li>
        </ul>
      </q-banner>
    </div>

    <!-- Dates Validation -->
    <div class="q-mb-md">
      <div class="text-subtitle1 q-mb-sm">Dates</div>
      <div v-if="validation.dates">
        <q-banner v-if="validation.dates.valid" class="bg-positive text-white">
          The data source covers the required experiment date range.
        </q-banner>
        <q-banner v-else class="bg-negative text-white">
          <div class="text-weight-bold">Date Range Mismatch:</div>
          <div>
            Experiment time frame: {{ validation.dates.required.start }} -
            {{ validation.dates.required.end }}
          </div>
          <div>
            Data source time frame: {{ validation.dates.actual.start }} -
            {{ validation.dates.actual.end }}
          </div>
        </q-banner>
      </div>
    </div>

    <!-- Geo Units Validation -->
    <div>
      <div class="text-subtitle1 q-mb-sm">Geo Units Mismatch</div>
      <div
        v-if="
          validation.geoUnits.designOnly.length === 0 &&
          validation.geoUnits.dataSourceOnly.length === 0
        "
      >
        <q-banner class="bg-positive text-white">
          Geo units are perfectly aligned between the design and the data source.
        </q-banner>
      </div>
      <div v-else>
        <q-banner class="bg-warning text-dark">
          <q-markup-table flat bordered separator="vertical" dense>
            <thead>
              <tr>
                <th>Geo Units Only in Design</th>
                <th>Geo Units Only in Data Source</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="index in Math.max(
                  validation.geoUnits.designOnly.length,
                  validation.geoUnits.dataSourceOnly.length,
                )"
                :key="index"
              >
                <td align="center">{{ validation.geoUnits.designOnly[index - 1] || '' }}</td>
                <td align="center">{{ validation.geoUnits.dataSourceOnly[index - 1] || '' }}</td>
              </tr>
            </tbody>
          </q-markup-table>
        </q-banner>
      </div>
    </div>
  </q-card>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import type { PropType } from 'vue';
import type { ValidationResult } from './models';

const props = defineProps({
  validation: {
    type: Object as PropType<ValidationResult>,
    required: true,
  },
});

const validationPassed = computed(() => {
  return (
    props.validation.metrics.missing.length === 0 &&
    (props.validation.dates ? props.validation.dates.valid : false)
  );
});

defineExpose({
  validationPassed,
});
</script>
