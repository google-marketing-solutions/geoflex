<template>
  <div>
    <q-card class="q-pa-md">
      <div class="row items-center q-mb-md">
        <div class="col">
          <div class="text-subtitle1">Found {{ designs.length }} designs</div>
        </div>
        <div class="col-auto">
          <q-select
            v-model="sortBy"
            :options="sortOptions"
            label="Sort by"
            outlined
            dense
            style="min-width: 200px"
          />
        </div>
      </div>

      <div v-for="(design, index) in sortedDesigns" :key="index" class="q-mb-lg">
        <q-card bordered>
          <q-card-section>
            <div class="row items-center">
              <div class="col">
                <div class="text-h6">Design #{{ index + 1 }} ({{ design.design.design_id }})</div>
                <div v-if="showMeta" class="text-caption">
                  Data source: {{ design.datasource_name }}
                </div>
                <div v-if="showMeta" class="text-caption">
                  Created: {{ new Date(design.timestamp).toLocaleString() }} (local time)
                </div>
                <div class="text-caption">Methodology: {{ design.design.methodology }}</div>
                <div class="text-caption">Duration: {{ design.design.runtime_weeks }} weeks</div>
              </div>
              <div class="col-auto">
                <q-btn-group flat>
                  <q-btn color="primary" icon="visibility" @click="viewDesign(design)" />
                  <q-btn color="positive" icon="download" @click="downloadDesign(design)" />
                  <q-btn
                    v-if="showUpload"
                    color="secondary"
                    icon="cloud_upload"
                    @click="emit('upload', design)"
                  >
                    <q-tooltip>Upload to Cloud Storage</q-tooltip>
                  </q-btn>
                  <q-btn
                    v-if="showAnalyze"
                    color="info"
                    icon="analytics"
                    @click="emit('analyze', design)"
                  >
                    <q-tooltip>Analyze Design</q-tooltip>
                  </q-btn>
                  <q-btn
                    v-if="showDelete"
                    color="negative"
                    icon="delete"
                    @click="emit('delete', design)"
                  >
                    <q-tooltip>Delete Design</q-tooltip>
                  </q-btn>
                </q-btn-group>
              </div>
            </div>
          </q-card-section>

          <q-separator />

          <q-card-section>
            <div class="row q-col-gutter-md">
              <div class="col-12 col-md-6">
                <div class="text-subtitle2">Statistical Properties</div>
                <div class="row q-col-gutter-md">
                  <div class="col-6">
                    <q-item dense>
                      <q-item-section>
                        <q-item-label caption
                          >MDE ({{
                            getMetricName(design.design.primary_metric) || 'Primary Metric'
                          }})</q-item-label
                        >
                        <q-item-label class="text-primary text-weight-bold">{{
                          design.mde ? design.mde.toFixed(2) + '%' : 'N/A'
                        }}</q-item-label>
                      </q-item-section>
                    </q-item>
                  </div>
                </div>
              </div>
            </div>

            <div class="q-mt-md">
              <div class="text-subtitle2">Groups</div>
              <div class="row q-col-gutter-md">
                <div
                  v-for="(geos, groupName) in getDesignGroups(design.design)"
                  :key="groupName"
                  class="col-12 col-md-4"
                >
                  <q-card flat bordered>
                    <q-card-section class="q-py-sm bg-primary text-white row items-center">
                      <div class="text-subtitle2 col">{{ groupName }} ({{ geos.length }} geos)</div>
                      <div class="col-auto">
                        <q-btn
                          flat
                          dense
                          round
                          icon="file_download"
                          color="white"
                          @click="exportDesignGroupToCsv(design.design, String(groupName))"
                        >
                          <q-tooltip>Export {{ groupName }} geos as CSV</q-tooltip>
                        </q-btn>
                      </div>
                    </q-card-section>
                    <q-card-section class="q-pa-sm">
                      <div class="geo-chips-container" style="max-height: 200px; overflow-y: auto">
                        <q-chip
                          v-for="geo in geos"
                          :key="geo"
                          :color="isFixedGeo(geo, String(groupName)) ? 'orange' : 'primary'"
                          text-color="white"
                          size="sm"
                          class="q-ma-xs"
                        >
                          {{ geo }}
                        </q-chip>
                      </div>
                    </q-card-section>
                  </q-card>
                </div>
              </div>
            </div>
          </q-card-section>
        </q-card>
      </div>
    </q-card>

    <!-- Design Detail Dialog -->
    <q-dialog v-model="designDetailDialog" maximized persistent>
      <q-card>
        <q-card-section class="row items-center">
          <div class="text-h6">Design Details</div>
          <q-space />
          <q-btn icon="close" flat round dense v-close-popup />
        </q-card-section>

        <q-separator />

        <q-card-section v-if="selectedDesign" class="q-pa-md">
          <div class="row q-col-gutter-md">
            <!-- Design metadata -->
            <div class="col-12 col-md-4">
              <q-card class="q-pa-md">
                <div class="text-subtitle1">Design Summary</div>
                <q-list dense>
                  <q-item>
                    <q-item-section>
                      <q-item-label caption>Primary Metric</q-item-label>
                      <q-item-label>{{
                        getMetricName(selectedDesign.design.primary_metric)
                      }}</q-item-label>
                    </q-item-section>
                  </q-item>
                  <q-item>
                    <q-item-section>
                      <q-item-label caption>Methodology</q-item-label>
                      <q-item-label>{{ selectedDesign.design.methodology }}</q-item-label>
                    </q-item-section>
                  </q-item>
                  <q-item>
                    <q-item-section>
                      <q-item-label caption>Runtime (Weeks)</q-item-label>
                      <q-item-label>{{ selectedDesign.design.runtime_weeks }}</q-item-label>
                    </q-item-section>
                  </q-item>
                  <q-item>
                    <q-item-section>
                      <q-item-label caption>Number of Cells</q-item-label>
                      <q-item-label>{{ selectedDesign.design.n_cells }}</q-item-label>
                    </q-item-section>
                  </q-item>
                  <q-item>
                    <q-item-section>
                      <q-item-label caption>Alpha</q-item-label>
                      <q-item-label>{{ selectedDesign.design.alpha }}</q-item-label>
                    </q-item-section>
                  </q-item>
                  <q-item>
                    <q-item-section>
                      <q-item-label caption>Hypothesis</q-item-label>
                      <q-item-label>{{
                        selectedDesign.design.alternative_hypothesis
                      }}</q-item-label>
                    </q-item-section>
                  </q-item>
                  <q-item>
                    <q-item-section>
                      <q-item-label caption>Budget</q-item-label>
                      <q-item-label
                        >{{ selectedDesign.design.experiment_budget.value }} ({{
                          selectedDesign.design.experiment_budget.budget_type
                        }})</q-item-label
                      >
                    </q-item-section>
                  </q-item>
                  <q-item>
                    <q-item-section>
                      <q-item-label caption>Methodology Parameters</q-item-label>
                      <q-markup-table flat bordered dense>
                        <tbody>
                          <tr
                            v-for="(val, key) in selectedDesign.design.methodology_parameters"
                            :key="key"
                          >
                            <td>{{ key }}</td>
                            <td>{{ val }}</td>
                          </tr>
                        </tbody>
                      </q-markup-table>
                    </q-item-section>
                  </q-item>
                  <q-item>
                    <q-item-section>
                      <q-item-label caption>Random Seed</q-item-label>
                      <q-item-label>{{ selectedDesign.design.random_seed }}</q-item-label>
                    </q-item-section>
                  </q-item>
                  <q-item>
                    <q-item-section>
                      <q-item-label caption>Effect Scope</q-item-label>
                      <q-item-label>{{ selectedDesign.design.effect_scope }}</q-item-label>
                    </q-item-section>
                  </q-item>
                  <q-item>
                    <q-item-section>
                      <q-item-label caption>Cell Volume Constraint</q-item-label>
                      <q-item-label>
                        <pre>{{
                          JSON.stringify(selectedDesign.design.cell_volume_constraint, null, 2)
                        }}</pre>
                      </q-item-label>
                    </q-item-section>
                  </q-item>
                </q-list>
              </q-card>
            </div>

            <!-- Statistics -->
            <div class="col-12 col-md-4">
              <q-card class="q-pa-md">
                <div class="text-subtitle1">Statistical Properties</div>
                <q-list dense>
                  <q-item>
                    <q-item-section>
                      <q-item-label caption
                        >MDE ({{
                          getMetricName(selectedDesign.design.primary_metric) || 'Primary Metric'
                        }})</q-item-label
                      >
                      <q-item-label class="text-h5">{{
                        selectedDesign.mde ? selectedDesign.mde.toFixed(1) + '%' : 'N/A'
                      }}</q-item-label>
                    </q-item-section>
                  </q-item>
                  <q-item>
                    <q-item-section>
                      <q-item-label caption>Duration</q-item-label>
                      <q-item-label class="text-h5"
                        >{{ selectedDesign.design.runtime_weeks }} wks</q-item-label
                      >
                    </q-item-section>
                  </q-item>
                </q-list>
              </q-card>
            </div>

            <!-- Evaluation Results -->
            <div class="col-12 col-md-4">
              <q-card class="q-pa-md">
                <div class="text-subtitle1">Evaluation Results</div>
                <q-list dense v-if="selectedDesign.design.evaluation_results">
                  <q-item>
                    <q-item-section>
                      <q-item-label caption>Valid Design</q-item-label>
                      <q-item-label>
                        <q-badge
                          :color="
                            selectedDesign.design.evaluation_results.is_valid_design
                              ? 'positive'
                              : 'negative'
                          "
                        >
                          {{
                            selectedDesign.design.evaluation_results.is_valid_design ? 'Yes' : 'No'
                          }}
                        </q-badge>
                      </q-item-label>
                    </q-item-section>
                  </q-item>
                  <q-item>
                    <q-item-section>
                      <q-item-label caption>Empirical Power</q-item-label>
                      <q-item-label
                        >{{
                          (
                            selectedDesign.design.evaluation_results.all_metric_results_per_cell[
                              selectedDesign.design.evaluation_results.primary_metric_name
                            ][0].empirical_power * 100
                          ).toFixed(2)
                        }}%</q-item-label
                      >
                    </q-item-section>
                  </q-item>
                  <q-item>
                    <q-item-section>
                      <q-item-label caption>Standard Error (Absolute)</q-item-label>
                      <q-item-label>{{
                        selectedDesign.design.evaluation_results.all_metric_results_per_cell[
                          selectedDesign.design.evaluation_results.primary_metric_name
                        ][0].standard_error_absolute_effect.toFixed(4)
                      }}</q-item-label>
                    </q-item-section>
                  </q-item>
                  <q-item>
                    <q-item-section>
                      <q-item-label caption>Standard Error (Relative)</q-item-label>
                      <q-item-label
                        >{{
                          (
                            selectedDesign.design.evaluation_results.all_metric_results_per_cell[
                              selectedDesign.design.evaluation_results.primary_metric_name
                            ][0].standard_error_relative_effect * 100
                          ).toFixed(2)
                        }}%</q-item-label
                      >
                    </q-item-section>
                  </q-item>
                  <q-item>
                    <q-item-section>
                      <q-item-label caption>Coverage (Relative)</q-item-label>
                      <q-item-label
                        >{{
                          (
                            selectedDesign.design.evaluation_results.all_metric_results_per_cell[
                              selectedDesign.design.evaluation_results.primary_metric_name
                            ][0].coverage_relative_effect * 100
                          ).toFixed(2)
                        }}%</q-item-label
                      >
                    </q-item-section>
                  </q-item>
                  <q-item
                    v-if="
                      selectedDesign.design.evaluation_results.failing_checks &&
                      selectedDesign.design.evaluation_results.failing_checks.length
                    "
                  >
                    <q-item-section>
                      <q-item-label caption>Failing Checks</q-item-label>
                      <q-item-label
                        v-for="check in selectedDesign.design.evaluation_results.failing_checks"
                        :key="check"
                      >
                        <q-chip dense color="negative" text-color="white">{{ check }}</q-chip>
                      </q-item-label>
                    </q-item-section>
                  </q-item>
                </q-list>
              </q-card>
            </div>
          </div>
        </q-card-section>

        <q-card-actions align="right">
          <q-btn color="primary" label="Download" @click="downloadDesign(selectedDesign)" />
          <q-btn color="grey" label="Close" v-close-popup />
        </q-card-actions>
      </q-card>
    </q-dialog>
  </div>
</template>

<script setup lang="ts">
import type { PropType } from 'vue';
import { ref, computed } from 'vue';
import { useQuasar, exportFile } from 'quasar';
import type { ExperimentDesign, AnyMetric, SavedDesign } from 'src/components/models';

const props = defineProps({
  designs: {
    type: Array as PropType<SavedDesign[]>,
    required: true,
  },
  showUpload: {
    type: Boolean,
    default: false,
  },
  showAnalyze: {
    type: Boolean,
    default: false,
  },
  showDelete: {
    type: Boolean,
    default: false,
  },
  showMeta: {
    type: Boolean,
    default: false,
  },
  fixedGeos: {
    type: Object as PropType<Record<string, string[]>>,
    default: () => ({}),
  },
});

const emit = defineEmits<{
  (e: 'analyze', design: SavedDesign): void;
  (e: 'upload', design: SavedDesign): void;
  (e: 'delete', design: SavedDesign): void;
}>();

const $q = useQuasar();

const sortOptions = [
  { label: 'Lowest MDE', value: 'mde' },
  { label: 'Shortest Duration', value: 'duration' },
  { label: 'Most Recent', value: 'recency' },
];

const sortBy = ref(sortOptions[0]);

const sortedDesigns = computed<SavedDesign[]>(() => {
  const designs = [...props.designs];
  const field = sortBy.value.value;

  if (field === 'mde') {
    return designs.sort((a, b) => (a.mde ?? Infinity) - (b.mde ?? Infinity));
  } else if (field === 'duration') {
    return designs.sort((a, b) => a.design.runtime_weeks - b.design.runtime_weeks);
  } else if (field === 'recency') {
    return designs.sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
    );
  }
  return designs;
});

// Design detail dialog
const designDetailDialog = ref(false);
const selectedDesign = ref<SavedDesign | null>(null);

function viewDesign(design: SavedDesign) {
  selectedDesign.value = design;
  designDetailDialog.value = true;
}

function downloadDesign(design: SavedDesign | null) {
  if (!design) return;
  const filename = `geo-test-design-${design.design.design_id || Date.now()}.json`;
  exportFile(filename, JSON.stringify(design.design, null, 2));
}

function getDesignGroups(design: ExperimentDesign) {
  const groups: { [key: string]: string[] } = {
    Control: design.geo_assignment.control,
  };
  design.geo_assignment.treatment.forEach((treatmentGroup, index) => {
    const groupName = `Treatment ${String.fromCharCode(65 + index)}`;
    groups[groupName] = treatmentGroup;
  });
  return groups;
}

function exportDesignGroupToCsv(design: ExperimentDesign, groupName: string) {
  const groups = getDesignGroups(design);
  const geos = groups[groupName];

  if (!geos || geos.length === 0) {
    $q.notify({
      color: 'info',
      message: `Group '${groupName}' has no geo units to export.`,
      icon: 'info',
    });
    return;
  }

  let csvContent = 'geo_id\n';
  csvContent += geos.join('\n');

  const designId = design.design_id || 'unknown_design';
  const safeGroupName = String(groupName)
    .replace(/[^a-z0-9_]/gi, '_')
    .toLowerCase();
  const filename = `${designId}_${safeGroupName}.csv`;

  const status = exportFile(filename, csvContent, 'text/csv;charset=utf-8;');

  if (status !== true) {
    $q.notify({
      color: 'negative',
      message: 'CSV export failed. Please try again.',
      icon: 'warning',
    });
  } else {
    $q.notify({
      color: 'positive',
      message: `Successfully exported ${filename}`,
      icon: 'check_circle',
    });
  }
}

function getMetricName(metric: AnyMetric | undefined): string {
  if (!metric) return '';
  if (typeof metric === 'string') {
    return metric;
  }
  return metric.name;
}

function isFixedGeo(geo: string, groupName: string): boolean {
  if (!props.fixedGeos) return false;

  if (groupName === 'Control' && props.fixedGeos.control?.includes(geo)) {
    return true;
  }

  if (groupName.startsWith('Treatment')) {
    let groupIndex = -1;
    if (groupName === 'Treatment') {
      groupIndex = 0; // Single treatment group
    } else if (groupName.startsWith('Treatment ')) {
      const letter = groupName.split(' ')[1];
      if (letter && letter.length === 1) {
        groupIndex = letter.charCodeAt(0) - 65; // 'A' -> 0, 'B' -> 1
      }
    }

    if (
      groupIndex !== -1 &&
      props.fixedGeos.treatment &&
      props.fixedGeos.treatment[groupIndex]?.includes(geo)
    ) {
      return true;
    }
  }

  return false;
}
</script>

<style scoped>
.geo-chips-container {
  display: flex;
  flex-wrap: wrap;
}
pre {
  white-space: pre-wrap;
  word-wrap: break-word;
}
</style>
