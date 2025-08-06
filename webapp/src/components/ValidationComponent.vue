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
  return props.validation.metrics.missing.length === 0;
});

defineExpose({
  validationPassed,
});
</script>
