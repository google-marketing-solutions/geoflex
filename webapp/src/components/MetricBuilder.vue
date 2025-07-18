<template>
  <q-card flat bordered class="q-pa-md">
    <div class="row q-col-gutter-md">
      <div class="col-12 col-md-4">
        <q-select
          v-model="selectedMetricType"
          :options="metricTypeOptions"
          label="Metric Type"
          outlined
          dense
          emit-value
          map-options
          @update:model-value="resetMetric"
        />
      </div>
      <div class="col-12 col-md-8">
        <!-- Simple Metric -->
        <div v-if="selectedMetricType === 'simple'">
          <q-select
            :model-value="modelValue"
            @update:model-value="(val) => emit('update:modelValue', val)"
            :options="metricColumns"
            label="Conversion Metric"
            outlined
            dense
            hint="Select a column to use as the primary metric"
          />
        </div>

        <!-- Custom Metric -->
        <div v-if="selectedMetricType === 'custom'" class="q-gutter-y-md">
          <q-select
            v-model="internalMetric.name"
            :options="metricColumns"
            label="Metric Column"
            outlined
            dense
          />
          <q-select
            v-model="internalMetric.cost_column"
            :options="[costColumn]"
            :disable="!costColumn"
            label="Cost Column"
            outlined
            dense clearable
            hint="Required for cost-per-metric or metric-per-cost"
          />
          <q-toggle v-model="internalMetric.metric_per_cost" label="Metric per Cost (e.g., ROAS)" />
          <q-toggle v-model="internalMetric.cost_per_metric" label="Cost per Metric (e.g., CPA)" />
        </div>
        <q-banner v-if="showCostWarning" dense inline-actions class="text-white bg-warning q-my-md">
          <template v-slot:avatar>
            <q-icon name="warning" color="white" />
          </template>
          A cost column is required for this metric type but is not available in the selected data source.
        </q-banner>

        <!-- iROAS Metric -->
        <div v-if="selectedMetricType === 'iroas'" class="q-gutter-y-md">
          <q-input v-model="internalMetric.name" label="Metric Name" outlined dense readonly />
          <q-select
            v-model="internalMetric.return_column"
            :options="metricColumns"
            label="Return Column (e.g., revenue)"
            outlined
            dense
          />
          <q-select
            v-model="internalMetric.cost_column"
            :options="[costColumn]"
            :disable="!costColumn"
            label="Cost Column"
            outlined
            dense
          />
        </div>

        <!-- CPiA Metric -->
        <div v-if="selectedMetricType === 'cpia'" class="q-gutter-y-md">
          <q-input v-model="internalMetric.name" label="Metric Name" outlined dense readonly />
          <q-select
            v-model="internalMetric.conversions_column"
            :options="metricColumns"
            label="Conversions Column"
            outlined
            dense
          />
          <q-select
            v-model="internalMetric.cost_column"
            :options="[costColumn]"
            :disable="!costColumn"
            label="Cost Column"
            outlined
            dense
          />
        </div>
      </div>
    </div>
  </q-card>
</template>

<script setup lang="ts">
import { ref, watch, computed } from 'vue';
import type { Metric, AnyMetric } from './models';

const props = defineProps<{
  modelValue: AnyMetric;
  metricColumns: string[];
  costColumn?: string;
}>();

const emit = defineEmits(['update:modelValue']);

const selectedMetricType = ref<'simple' | 'custom' | 'iroas' | 'cpia'>('simple');

const metricTypeOptions = [
  { label: 'Simple Metric', value: 'simple' },
  { label: 'Custom Metric', value: 'custom' },
  { label: 'iROAS (Incremental ROAS)', value: 'iroas' },
  { label: 'CPiA (Cost Per Incremental Acquisition)', value: 'cpia' },
];

// This computed property is the key to simplifying the component.
// It provides a mutable object for the template to bind to,
// and emits updates when changed.
const internalMetric = computed<Metric>({
  get() {
    if (typeof props.modelValue === 'object') {
      return props.modelValue;
    }
    // Provide a default structure for the template bindings to prevent errors
    // when the modelValue is a simple string.
    return { name: '', type: 'custom' };
  },
  set(newValue) {
    emit('update:modelValue', newValue);
  },
});

const showCostWarning = computed(() => {
  const isCostMetric =
    selectedMetricType.value === 'iroas' ||
    selectedMetricType.value === 'cpia' ||
    (selectedMetricType.value === 'custom' &&
      (internalMetric.value.metric_per_cost || internalMetric.value.cost_per_metric));
  return isCostMetric && !props.costColumn;
});

function resetMetric(newType: 'simple' | 'custom' | 'iroas' | 'cpia') {
  if (newType === 'simple') {
    emit('update:modelValue', props.metricColumns[0] || '');
  } else {
    const baseMetric: Metric = { name: '', type: newType };
    if (newType === 'iroas') {
      baseMetric.name = 'iROAS';
      baseMetric.cost_column = props.costColumn;
    } else if (newType === 'cpia') {
      baseMetric.name = 'CPiA';
      baseMetric.cost_column = props.costColumn;
    }
    emit('update:modelValue', baseMetric);
  }
}

// Watch the modelValue to update the selectedMetricType
watch(
  () => props.modelValue,
  (newValue) => {
    if (typeof newValue === 'string') {
      selectedMetricType.value = 'simple';
    } else if (newValue) {
      selectedMetricType.value = newValue.type;
    }
  },
  { immediate: true }
);

// Watch the toggles to enforce mutual exclusivity and set cost column
watch(
  () => internalMetric.value.metric_per_cost,
  (newValue) => {
    if (newValue) {
      internalMetric.value.cost_per_metric = false;
      internalMetric.value.cost_column = props.costColumn;
    }
  }
);

watch(
  () => internalMetric.value.cost_per_metric,
  (newValue) => {
    if (newValue) {
      internalMetric.value.metric_per_cost = false;
      internalMetric.value.cost_column = props.costColumn;
    }
  }
);
</script>
