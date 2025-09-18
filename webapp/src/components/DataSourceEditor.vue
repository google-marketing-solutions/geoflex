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
  <div class="q-pa-md">
    <q-stepper v-model="step" vertical color="primary" animated>
      <!-- Step 1: Basic Information -->
      <q-step :name="1" title="Basic Information" icon="settings" :done="step > 1">
        <q-card flat>
          <q-card-section>
            <div class="row q-col-gutter-md">
              <div class="col-12 col-md-6">
                <q-input
                  v-model="currentDataSource.name"
                  label="Data Source Name *"
                  filled
                  :rules="[(val) => !!val || 'Name is required']"
                />
              </div>

              <div class="col-12 col-md-6" v-if="!isEditing">
                <q-select
                  v-model="currentDataSource.sourceType"
                  :options="sourceTypeOptions"
                  label="Data Source Type *"
                  filled
                  emit-value
                  map-options
                />
              </div>

              <div class="col-12">
                <q-input
                  v-model="currentDataSource.description"
                  label="Description"
                  type="textarea"
                  hint="Optional description"
                  filled
                  autogrow
                />
              </div>
            </div>
          </q-card-section>
          <q-card-section v-if="!isEditing">
            <div v-if="currentDataSource.sourceType === DataSourceType.Internal">
              <p class="text-body1 q-mb-md">
                Upload a CSV file containing your geo-attributed time series data.
              </p>

              <q-file
                v-model="file"
                label="Select CSV File"
                filled
                counter
                accept=".csv"
                @update:model-value="handleFileUpload"
                :loading="loading"
              >
                <template v-slot:prepend>
                  <q-icon name="cloud_upload" />
                </template>
                <template v-slot:append v-if="file">
                  <q-icon name="close" @click.stop="clearFile" class="cursor-pointer" />
                </template>
                <template v-slot:hint>
                  CSV files with geo units, date units, and conversion metrics
                </template>
              </q-file>
            </div>

            <div v-else>
              <p class="text-body1 q-mb-md">Fetch data from an external data source.</p>

              <div class="row q-col-gutter-md">
                <div class="col-12">
                  <q-input
                    v-model="externalSourceUrl"
                    label="External Source URL (Google Sheets or similar) *"
                    filled
                    :rules="[(val) => !!val || 'URL is required']"
                  >
                    <template v-slot:prepend>
                      <q-icon name="link" />
                    </template>
                  </q-input>
                </div>
              </div>

              <q-btn
                color="primary"
                label="Fetch data"
                class="q-mt-md"
                @click="fetchExternalData"
                :disable="!externalSourceUrl"
                :loading="loadingExternalData"
              />
            </div>
          </q-card-section>
        </q-card>

        <q-stepper-navigation>
          <q-btn
            color="primary"
            label="Continue"
            @click="step = 2"
            :disable="!canContinueFromImport"
          />
          <q-btn flat color="primary" label="Cancel" class="q-ml-sm" @click="onCancel" />
        </q-stepper-navigation>
      </q-step>

      <!-- Step 2: Data Preview and Column Configuration -->
      <q-step :name="2" title="Data Preview and Column Configuration" icon="tune" :done="step > 2">
        <q-card flat>
          <q-card-section>
            <div v-if="rawData.length && columns.length">
              <div class="row q-col-gutter-md">
                <!-- Column Configuration -->
                <div class="col-12 col-md-6">
                  <div class="text-h6 q-mb-md">Column Configuration</div>
                  <q-list bordered separator>
                    <q-item v-for="col in columns" :key="col.name">
                      <q-item-section>
                        <q-item-label>{{ col.name }}</q-item-label>
                      </q-item-section>
                      <q-item-section side>
                        <q-select
                          v-model="columnRoles[col.name]"
                          :options="columnRoleOptions"
                          dense
                          outlined
                          @update:model-value="handleColumnRoleChange(col.name, $event)"
                        />
                      </q-item-section>
                    </q-item>
                  </q-list>

                  <div v-if="!isColumnConfigValid" class="text-negative text-caption q-mt-sm">
                    Please select at least one Geo, one Date, and one Metric column
                  </div>
                </div>

                <div class="col-12 col-md-4 offset-md-1">
                  <q-card class="bg-primary text-white q-mt-lg">
                    <q-card-section>
                      <div class="text-h6">Total Rows</div>
                      <div class="text-h4">{{ rawData.length }}</div>
                    </q-card-section>
                  </q-card>
                  <q-card class="bg-secondary text-white q-mt-lg">
                    <q-card-section>
                      <div class="text-h6">Geo Units</div>
                      <div class="text-h4">{{ geoUnits.length }}</div>
                    </q-card-section>
                  </q-card>
                  <q-card class="bg-accent text-white q-mt-lg">
                    <q-card-section>
                      <div class="text-h6">Date Range</div>
                      <div class="text-h4">
                        {{ uniqueDates.length }}
                        <span class="text-subtitle1"
                          >({{ formatDate(uniqueDates[0]) }} —
                          {{ formatDate(uniqueDates[uniqueDates.length - 1]) }})</span
                        >
                      </div>
                    </q-card-section>
                  </q-card>
                </div>
              </div>

              <div class="row q-col-gutter-md q-mt-lg">
                <div class="col-12 col-md-4"></div>

                <div class="col-12 col-md-4"></div>

                <div class="col-12 col-md-4"></div>
              </div>
              <div class="row">
                <!-- Data Preview -->
                <div class="col">
                  <div class="text-h6 q-mb-md">Data Preview</div>
                  <q-table
                    ref="dataPreviewTable"
                    :rows="processedRawData"
                    :columns="dataPreviewColumns"
                    row-key="__index"
                    dense
                    :pagination="{ rowsPerPage: 10 }"
                  >
                    <template v-slot:top>
                      <q-input v-model="search" debounce="300" placeholder="Search" dense filled>
                        <template v-slot:append>
                          <q-icon name="search" />
                        </template>
                      </q-input>
                    </template>
                    <template v-slot:body="props">
                      <q-tr
                        :props="props"
                        :class="{ 'highlighted-row': props.row.__index === highlightedRow }"
                      >
                        <q-td v-for="col in props.cols" :key="col.name" :props="props">
                          <div v-if="col.name === 'status'">
                            <q-icon
                              v-if="props.row.errors && props.row.errors.length > 0"
                              name="warning"
                              color="warning"
                            >
                              <q-tooltip>
                                <div v-for="(error, index) in props.row.errors" :key="index">
                                  {{ error }}
                                </div>
                              </q-tooltip>
                            </q-icon>
                          </div>
                          <div v-else>
                            {{ col.value }}
                          </div>
                        </q-td>
                      </q-tr>
                    </template>
                  </q-table>
                </div>
              </div>

              <!-- Validation Errors (Data Quality Issues) -->
              <div class="q-my-md">
                <q-btn
                  label="Validate Data"
                  color="secondary"
                  @click="triggerValidation"
                  :loading="validateDataLoading"
                  icon="rule"
                />
              </div>

              <div v-if="dataQualityIssues.length > 0" class="q-mt-md">
                <q-card flat bordered>
                  <q-card-section>
                    <div class="text-h6 q-mb-sm">Data Quality Issues</div>
                    <q-banner class="bg-info text-white">
                      Please fix validation errors before continue. Found
                      {{ dataQualityIssues.filter((i) => i.isFixable).length }} fixable and
                      {{ dataQualityIssues.filter((i) => !i.isFixable).length }} unfixable errors.
                    </q-banner>
                    <q-table
                      :rows="dataQualityIssues"
                      :columns="issueColumns"
                      row-key="rowIndex"
                      dense
                      flat
                      :pagination="{ rowsPerPage: 10 }"
                      :filter="issueFilter"
                      :filter-method="filterIssues"
                    >
                      <template v-slot:top>
                        <div class="q-gutter-sm row">
                          <q-select
                            v-model="issueFilter.type"
                            :options="issueTypeOptions"
                            label="Filter by Type"
                            emit-value
                            map-options
                            clearable
                            dense
                            outlined
                            style="min-width: 200px"
                          />
                          <q-select
                            v-model="issueFilter.fixable"
                            :options="[
                              { label: 'All', value: null },
                              { label: 'Fixable', value: true },
                              { label: 'Not Fixable', value: false },
                            ]"
                            label="Filter by Fixable"
                            emit-value
                            map-options
                            clearable
                            dense
                            outlined
                            style="min-width: 200px"
                          />
                        </div>
                      </template>
                      <template v-slot:body-cell-rowIndex="props">
                        <q-td :props="props" v-if="props.row.row">
                          <q-btn
                            flat
                            dense
                            color="primary"
                            @click="scrollToRow(props.row.rowIndex)"
                            :label="props.row.rowIndex + 1"
                          >
                            <q-tooltip v-if="props.row.row">
                              <div v-for="(value, key) in props.row.row" :key="key">
                                <strong>{{ key }}</strong
                                >: {{ value }}
                              </div>
                            </q-tooltip>
                          </q-btn>
                        </q-td>
                        <q-td v-else class="text-right">---</q-td>
                      </template>
                    </q-table>
                    <div class="q-mt-sm">
                      <q-btn
                        label="Attempt to Fix Issues"
                        color="primary"
                        @click="openFixDialog"
                        :loading="isFixingIssues"
                        icon="auto_fix_high"
                      />
                    </div>
                  </q-card-section>
                </q-card>
              </div>
            </div>
          </q-card-section>
        </q-card>

        <q-stepper-navigation>
          <q-btn color="primary" label="Continue" @click="step = 3" :disable="!isFormValid" />
          <q-btn flat color="primary" label="Back" class="q-mr-sm" @click="step = 1" />
          <q-btn flat color="primary" label="Cancel" class="q-ml-sm" @click="onCancel" />
        </q-stepper-navigation>
      </q-step>

      <!-- Step 3: Save -->
      <q-step :name="3" title="Save" icon="save">
        <q-card flat>
          <q-card-section>
            <div class="text-h6 q-mb-md">Review and Save</div>
            <p class="text-body1">
              Please review your data source configuration before saving. The data source will be
              saved with the following settings:
            </p>
            <ul class="q-mt-md">
              <li>Name: {{ currentDataSource.name }}</li>
              <li>Geo Column: {{ currentDataSource.columns.geoColumn }}</li>
              <li>Date Column: {{ currentDataSource.columns.dateColumn }}</li>
              <li>Metric Columns: {{ currentDataSource.columns.metricColumns.join(', ') }}</li>
              <li v-if="currentDataSource.columns.costColumn">
                Cost Column: {{ currentDataSource.columns.costColumn }}
              </li>
              <li>Data: {{ rawData.length }} rows</li>
            </ul>
          </q-card-section>
        </q-card>

        <q-stepper-navigation>
          <q-btn
            color="primary"
            label="Save Data Source"
            @click="saveDataSource"
            :loading="saving"
            :disable="!isFormValid"
          />
          <q-btn flat color="primary" label="Back" class="q-mr-sm" @click="step = 3" />
          <q-btn flat color="primary" label="Cancel" class="q-ml-sm" @click="onCancel" />
        </q-stepper-navigation>
      </q-step>
    </q-stepper>

    <!-- Error Dialog -->
    <q-dialog v-model="errorDialog" persistent>
      <q-card>
        <q-card-section class="row items-center">
          <q-avatar icon="error" color="negative" text-color="white" />
          <span class="q-ml-sm">{{ errorMessage }}</span>
        </q-card-section>
        <q-card-actions align="right">
          <q-btn flat label="Dismiss" color="primary" v-close-popup />
        </q-card-actions>
      </q-card>
    </q-dialog>

    <!-- Fix Selection Dialog -->
    <q-dialog v-model="showFixSelectionDialog" persistent>
      <q-card style="min-width: 350px">
        <q-card-section>
          <div class="text-h6">Select Fixes to Apply</div>
        </q-card-section>

        <q-card-section class="q-pt-none">
          <q-option-group
            v-model="fixSelectionArray"
            :options="fixSelectionOptions"
            type="checkbox"
            dense
          />
        </q-card-section>

        <q-card-actions align="right">
          <q-btn flat label="Cancel" color="primary" v-close-popup />
          <q-btn
            label="Apply Selected Fixes"
            color="primary"
            @click="applySelectedFixes"
            :loading="isFixingIssues"
          />
        </q-card-actions>
      </q-card>
    </q-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, nextTick } from 'vue';
import { QTable, useQuasar } from 'quasar';
import { formatDate } from 'src/helpers/utils';

enum DataQualityIssueType {
  MissingGeo = 'MissingGeo',
  MissingDate = 'MissingDate',
  MissingCost = 'MissingCost',
  MissingMetric = 'MissingMetric',
  InvalidDateFormat = 'InvalidDateFormat',
  NonNumericMetric = 'NonNumericMetric',
  DateGap = 'DateGap',
  DuplicateGeoDate = 'DuplicateGeoDate',
}

interface DataQualityIssue {
  type: DataQualityIssueType;
  message: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  row?: any;
  rowIndex?: number;
  isFixable: boolean;
}
import { parse } from 'csv-parse/browser/esm';
import type { DataSource } from 'stores/datasources';
import { useDataSourcesStore, DataSourceType } from 'stores/datasources';
import { getApiUi } from 'src/boot/axios';
import { assertIsError } from 'src/helpers/utils';

interface Props {
  dataSource?: DataSource | null;
}

const props = withDefaults(defineProps<Props>(), {
  dataSource: null,
});

const emit = defineEmits<{
  (e: 'saved', dataSource: DataSource): void;
  (e: 'canceled'): void;
}>();

const $q = useQuasar();
const dataSourcesStore = useDataSourcesStore();

// Step management
const step = ref(1);
const isEditing = computed(() => !!props.dataSource);

// Form data
const currentDataSource = ref<DataSource>(dataSourcesStore.createEmptyDataSource());

// UI state
const file = ref<File | null>(null);
const loading = ref(false);
const saving = ref(false);
const errorDialog = ref(false);
const errorMessage = ref('');
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const rawData = ref<any[]>([]);
const columns = ref<{ name: string; label: string; field: string }[]>([]);
const dataPreviewTable = ref<QTable | null>(null);
const highlightedRow = ref<number | null>(null);
const search = ref('');
const loadingExternalData = ref(false);
const externalSourceUrl = ref('');
const isFixingIssues = ref(false);
const validateDataLoading = ref(false);
const showFixSelectionDialog = ref(false);

interface FixSelection {
  removeMissingCritical: boolean;
  fillMissingMetrics: boolean;
  fillDateGaps: boolean;
  deduplicateRows: boolean;
}
const fixSelection = ref<FixSelection>({
  removeMissingCritical: true,
  fillMissingMetrics: true,
  fillDateGaps: true,
  deduplicateRows: true,
});

// For q-option-group which expects an array for multiple checkboxes
const fixSelectionArray = computed({
  get: () => {
    const arr = [];
    if (fixSelection.value.removeMissingCritical) arr.push('removeMissingCritical');
    if (fixSelection.value.fillMissingMetrics) arr.push('fillMissingMetrics');
    if (fixSelection.value.fillDateGaps) arr.push('fillDateGaps');
    if (fixSelection.value.deduplicateRows) arr.push('deduplicateRows');
    return arr;
  },
  set: (val: string[]) => {
    fixSelection.value.removeMissingCritical = val.includes('removeMissingCritical');
    fixSelection.value.fillMissingMetrics = val.includes('fillMissingMetrics');
    fixSelection.value.fillDateGaps = val.includes('fillDateGaps');
    fixSelection.value.deduplicateRows = val.includes('deduplicateRows');
  },
});

const availableFixTypes = computed(() => {
  const types = {
    removeMissingCritical: false,
    fillMissingMetrics: false,
    fillDateGaps: false,
    deduplicateRows: false,
  };

  const currentIssues: DataQualityIssue[] = dataQualityIssues.value;

  if (
    currentIssues.some(
      (issue) =>
        issue.type === DataQualityIssueType.MissingGeo ||
        issue.type === DataQualityIssueType.MissingDate ||
        issue.type === DataQualityIssueType.MissingCost,
    )
  ) {
    types.removeMissingCritical = true;
  }

  if (currentIssues.some((issue) => issue.type === DataQualityIssueType.MissingMetric)) {
    types.fillMissingMetrics = true;
  }

  if (currentIssues.some((issue) => issue.type === DataQualityIssueType.DateGap)) {
    types.fillDateGaps = true;
  }

  if (currentIssues.some((issue) => issue.type === DataQualityIssueType.DuplicateGeoDate)) {
    types.deduplicateRows = true;
  }

  if (currentIssues.some((issue) => !issue.isFixable)) {
    types.removeMissingCritical = false;
    types.fillMissingMetrics = false;
    types.fillDateGaps = false;
    types.deduplicateRows = false;
  }

  return types;
});

const fixSelectionOptions = computed(() => [
  {
    label: 'Remove rows with missing Geo/Date/Cost',
    value: 'removeMissingCritical',
    disable: !availableFixTypes.value.removeMissingCritical,
  },
  {
    label: 'Fill missing metric values with 0',
    value: 'fillMissingMetrics',
    disable: !availableFixTypes.value.fillMissingMetrics,
  },
  {
    label: 'Fill date gaps with empty entries',
    value: 'fillDateGaps',
    disable: !availableFixTypes.value.fillDateGaps,
  },
  {
    label: 'Remove duplicate Geo/Date rows',
    value: 'deduplicateRows',
    disable: !availableFixTypes.value.deduplicateRows,
  },
]);

// Select options
const sourceTypeOptions = [
  { label: 'Internal (Upload CSV)', value: DataSourceType.Internal },
  { label: 'External (Google Sheets, etc.)', value: DataSourceType.External },
];

const columnRoles = ref<Record<string, ColumnRole>>({});

enum ColumnRole {
  Geo = 'Geo',
  Date = 'Date',
  Metric = 'Metric',
  Cost = 'Cost',
  Ignore = 'Ignore',
}

const columnRoleOptions = [
  { label: 'Geo', value: ColumnRole.Geo },
  { label: 'Date', value: ColumnRole.Date },
  { label: 'Metric', value: ColumnRole.Metric },
  { label: 'Cost', value: ColumnRole.Cost },
  { label: 'Ignore', value: ColumnRole.Ignore },
];

const dataPreviewColumns = computed(() => {
  return [
    { name: 'status', label: 'Status', field: 'status' },
    ...columns.value.map((col) => ({
      ...col,
      sortable: true,
    })),
  ];
});

// extended datasource rows array with validation errors grouped by row
const processedRawData = computed(() => {
  const issuesByRow = new Map<number, string[]>();
  dataQualityIssues.value.forEach((issue) => {
    if (issue.rowIndex !== undefined) {
      if (!issuesByRow.has(issue.rowIndex)) {
        issuesByRow.set(issue.rowIndex, []);
      }
      issuesByRow.get(issue.rowIndex)?.push(issue.message);
    }
  });

  return rawData.value.map((row, index) => ({
    ...row,
    errors: issuesByRow.get(index) || [],
  }));
});

const geoUnits = ref<string[]>([]);
const uniqueDates = ref<string[]>([]);

const dataQualityIssues = computed((): DataQualityIssue[] => {
  const issues: DataQualityIssue[] = [];
  if (rawData.value.length === 0) return issues;

  const { dateColumn, geoColumn, metricColumns, costColumn } = currentDataSource.value.columns;

  // Phase 1: Row-level validation
  for (let i = 0; i < rawData.value.length; i++) {
    const row = rawData.value[i];

    // Date Column
    if (dateColumn) {
      const dateVal = row[dateColumn];
      if (dateVal === undefined || dateVal === null || String(dateVal).trim() === '') {
        issues.push({
          type: DataQualityIssueType.MissingDate,
          message: `Missing date value.`,
          row,
          rowIndex: i,
          isFixable: true,
        });
      } else {
        const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
        if (
          !dateRegex.test(String(dateVal)) ||
          isNaN(new Date(dateVal as string | number | Date).getTime())
        ) {
          issues.push({
            type: DataQualityIssueType.InvalidDateFormat,
            message: `Invalid date format: '${String(dateVal)}'. Expected YYYY-MM-DD.`,
            row,
            rowIndex: i,
            isFixable: false,
          });
        }
      }
    }

    // Geo Column
    if (geoColumn) {
      const geoVal = row[geoColumn];
      if (geoVal === undefined || geoVal === null || String(geoVal).trim() === '') {
        issues.push({
          type: DataQualityIssueType.MissingGeo,
          message: `Missing geo value.`,
          row,
          rowIndex: i,
          isFixable: true,
        });
      }
    }

    // Metric Columns
    if (metricColumns.length > 0) {
      for (const metric of metricColumns) {
        const metricVal = row[metric];
        if (metricVal === undefined || metricVal === null || metricVal === '') {
          issues.push({
            type: DataQualityIssueType.MissingMetric,
            message: `Missing value in metric column '${metric}'.`,
            row,
            rowIndex: i,
            isFixable: true,
          });
        } else if (
          typeof metricVal !== 'number' &&
          (typeof metricVal !== 'string' || isNaN(Number(metricVal)))
        ) {
          issues.push({
            type: DataQualityIssueType.NonNumericMetric,
            message: `Non-numeric value in metric column '${metric}': '${metricVal}'.`,
            row,
            rowIndex: i,
            isFixable: false,
          });
        }
      }
    }

    // Cost Column
    if (costColumn) {
      const costVal = row[costColumn];
      if (!costVal) {
        issues.push({
          type: DataQualityIssueType.MissingCost,
          message: `Missing value in Cost column '${costColumn}'.`,
          row,
          rowIndex: i,
          isFixable: true,
        });
      } else if (
        typeof costVal !== 'number' &&
        (typeof costVal !== 'string' || isNaN(Number(costVal)))
      ) {
        issues.push({
          type: DataQualityIssueType.NonNumericMetric,
          message: `Non-numeric value in Cost column '${costColumn}': '${costVal}'.`,
          row,
          rowIndex: i,
          isFixable: false,
        });
      }
    }
  }

  // If there are unfixable format errors, return only those.
  const unfixableIssues = issues.filter((issue) => !issue.isFixable);
  if (unfixableIssues.length > 0) {
    return unfixableIssues;
  }

  // Phase 2: Dataset-level validation (gaps, duplicates)
  if (dateColumn && geoColumn) {
    const uniqueDates = [...new Set(rawData.value.map((row) => row[dateColumn]))]
      .map((d) => new Date(d))
      .sort((a, b) => a.getTime() - b.getTime());

    // Date Gap check
    if (uniqueDates.length > 1) {
      const firstDate = new Date(uniqueDates[0]);
      const lastDate = new Date(uniqueDates[uniqueDates.length - 1]);

      // Create a Set for O(1) lookup of existing dates
      const existingDates = new Set(uniqueDates.map((date) => date.toDateString()));

      const missingDates: Date[] = [];
      const currentDate = new Date(firstDate);
      while (currentDate <= lastDate) {
        if (!existingDates.has(currentDate.toDateString())) {
          missingDates.push(new Date(currentDate)); // it's important to clone the object
        }
        // Move to next day
        currentDate.setDate(currentDate.getDate() + 1);
      }

      if (missingDates.length) {
        issues.push({
          type: DataQualityIssueType.DateGap,
          message: `Date gaps detected. ${missingDates.length} day(s) missing: ${missingDates
            .slice(0, 11)
            .map((d) => d.toDateString())
            .join(', ')}`,
          isFixable: true,
        });
      }
    }

    // Duplicate Geo-Date check
    const geoDateCounts = new Map<string, number[]>();
    for (let i = 0; i < rawData.value.length; i++) {
      const row = rawData.value[i];
      const key = `${row[geoColumn]}|${row[dateColumn]}`;
      if (!geoDateCounts.has(key)) {
        geoDateCounts.set(key, []);
      }
      geoDateCounts.get(key)?.push(i);
    }

    for (const [key, indices] of geoDateCounts.entries()) {
      if (indices.length > 1) {
        const [geo, date] = key.split('|');
        issues.push({
          type: DataQualityIssueType.DuplicateGeoDate,
          message: `Duplicate geo-date combination found for Geo '${geo}' on Date '${date}'.`,
          row: rawData.value[indices[0]], // Show the first occurrence
          rowIndex: indices[0],
          isFixable: true,
        });
      }
    }
  }

  return issues;
});

const issueColumns = [
  {
    name: 'rowIndex',
    label: 'Row',
    field: 'rowIndex',
    sortable: true,
    format: (val: number) => val + 1,
  },
  { name: 'type', label: 'Issue Type', field: 'type', sortable: true },
  {
    name: 'message',
    label: 'Message',
    field: 'message',
    sortable: true,
    style: 'white-space: normal;',
  },
  {
    name: 'isFixable',
    label: 'Fixable',
    field: 'isFixable',
    sortable: true,
    format: (val: boolean) => (val ? 'Yes' : 'No'),
  },
];

const issueFilter = ref({
  type: null,
  fixable: null,
});

const issueTypeOptions = Object.values(DataQualityIssueType).map((type) => ({
  label: type,
  value: type,
}));

function filterIssues(
  rows: readonly DataQualityIssue[],
  terms: { type: DataQualityIssueType | null; fixable: boolean | null },
) {
  let filteredRows = rows;

  if (terms.type) {
    filteredRows = filteredRows.filter((row) => row.type === terms.type);
  }

  if (terms.fixable !== null) {
    filteredRows = filteredRows.filter((row) => row.isFixable === terms.fixable);
  }

  return filteredRows;
}

const canContinueFromImport = computed(() => {
  return currentDataSource.value.name && rawData.value.length > 0;
});

const isColumnConfigValid = computed(() => {
  return (
    !!currentDataSource.value.columns.geoColumn &&
    !!currentDataSource.value.columns.dateColumn &&
    currentDataSource.value.columns.metricColumns.length > 0
  );
});

const isFormValid = computed(() => {
  return (
    !!currentDataSource.value.name &&
    rawData.value.length > 0 &&
    isColumnConfigValid.value &&
    dataQualityIssues.value.length === 0
  );
});

async function triggerValidation() {
  validateDataLoading.value = true;
  await new Promise((resolve) => setTimeout(resolve, 200));
  const issuesCount = dataQualityIssues.value.length;

  if (issuesCount > 0) {
    $q.notify({
      type: 'warning',
      message: `Validation complete. ${issuesCount} issue(s) found. Please review the Data Quality Issues section.`,
      position: 'top',
      timeout: 3000,
    });
  } else {
    $q.notify({
      type: 'positive',
      message: 'Validation complete. No data quality issues detected.',
      position: 'top',
      timeout: 3000,
    });
  }
  validateDataLoading.value = false;
}

onMounted(async () => {
  // If editing, load existing data source
  const ds = props.dataSource;
  if (ds) {
    // editing an existing DS
    if (!props.dataSource.data) {
      loading.value = true;
      try {
        ds.data = await dataSourcesStore.loadDataSourceData(props.dataSource);
      } catch (error) {
        showError('Failed to load data source details: ' + error);
      } finally {
        loading.value = false;
      }
    }
    if (ds.data) {
      const data = ds.data;
      ds.data = undefined;
      currentDataSource.value = { ...ds };
      currentDataSource.value.data = data;
      ds.data = data;
    } else {
      currentDataSource.value = { ...ds };
    }

    // If there's data loaded, create columns for the editor
    if (currentDataSource.value.data && currentDataSource.value.data.rawRows?.length > 0) {
      rawData.value = currentDataSource.value.data.rawRows.map((row, index) => ({
        ...row,
        __index: index,
      }));

      // Create column definitions from the first row
      if (rawData.value.length > 0) {
        columns.value = Object.keys(rawData.value[0])
          .filter((key) => key !== '__index')
          .map((key) => ({
            name: key,
            label: key,
            field: key,
          }));
      }

      // Initialize column roles
      columnRoles.value = {};
      columns.value.forEach((col) => {
        columnRoles.value[col.name] = ColumnRole.Ignore;
      });

      // Set roles based on current configuration
      if (currentDataSource.value.columns.geoColumn) {
        columnRoles.value[currentDataSource.value.columns.geoColumn] = ColumnRole.Geo;
      }
      if (currentDataSource.value.columns.dateColumn) {
        columnRoles.value[currentDataSource.value.columns.dateColumn] = ColumnRole.Date;
      }
      if (currentDataSource.value.columns.costColumn) {
        columnRoles.value[currentDataSource.value.columns.costColumn] = ColumnRole.Cost;
      }
      currentDataSource.value.columns.metricColumns.forEach((metric) => {
        columnRoles.value[metric] = ColumnRole.Metric;
      });

      updateDataStats();
    }
  }
});

function showError(message: string) {
  errorMessage.value = message;
  errorDialog.value = true;
}

async function processData(data: Record<string, unknown>[]) {
  if (!data || data.length === 0) {
    throw new Error('No data found');
  }

  rawData.value = data.map((row, index) => ({
    ...row,
    __index: index,
  }));

  const firstRow = data[0];
  columns.value = Object.keys(firstRow)
    .filter((key) => key !== '__index')
    .map((key) => ({
      name: key,
      label: key,
      field: key,
    }));

  columnRoles.value = {};
  columns.value.forEach((col) => {
    columnRoles.value[col.name] = ColumnRole.Ignore;
  });

  await autoDetectColumns();
}

async function handleFileUpload() {
  if (!file.value) return;

  loading.value = true;

  try {
    const text = await file.value.text();

    // Use csv-parse to parse the CSV file
    const parsePromise = new Promise((resolve, reject) => {
      parse(
        text,
        {
          columns: true,
          skip_empty_lines: true,
          cast: true, // Attempt to auto-convert values to native types
          trim: true, // Trim whitespace
        },
        (err, records) => {
          if (err) reject(err);
          else resolve(records);
        },
      );
    });

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const parsedData = (await parsePromise) as any[];

    if (!parsedData || parsedData.length === 0) {
      throw new Error('The CSV file appears to be empty or has invalid format');
    }

    await processData(parsedData);

    // Set data source name from file if not already set
    if (!currentDataSource.value.name && file.value.name) {
      currentDataSource.value.name = file.value.name.replace(/\.[^/.]+$/, '');
    }
    currentDataSource.value.sourceLink = file.value.name;

    $q.notify({
      type: 'positive',
      message: 'File loaded successfully',
      position: 'top',
      timeout: 2000,
    });
  } catch (error) {
    assertIsError(error);
    showError(error.message || 'Failed to parse CSV file');
    clearFile();
  } finally {
    loading.value = false;
  }
}

function clearFile() {
  file.value = null;
  rawData.value = [];
  columns.value = [];
}

async function autoDetectColumns() {
  // Reset previous selections
  Object.keys(columnRoles.value).forEach((colName) => {
    columnRoles.value[colName] = ColumnRole.Ignore;
  });

  const colNames = columns.value.map((col) => col.name.toLowerCase());

  // Try to find geo column
  const geoKeywords = ['geo', 'state', 'dma', 'zipcode', 'country', 'region', 'city', 'location'];
  for (const keyword of geoKeywords) {
    const matchIndex = colNames.findIndex((name) => name.includes(keyword));
    if (matchIndex !== -1) {
      columnRoles.value[columns.value[matchIndex].name] = ColumnRole.Geo;
      break;
    }
  }

  // Try to find date column
  const dateKeywords = ['date', 'time', 'day', 'week', 'month', 'year', 'period'];
  for (const keyword of dateKeywords) {
    const matchIndex = colNames.findIndex((name) => name.includes(keyword));
    if (matchIndex !== -1) {
      columnRoles.value[columns.value[matchIndex].name] = ColumnRole.Date;
      break;
    }
  }

  // Try to find cost column
  const costKeywords = ['cost', 'spend', 'budget', 'expense', 'investment'];
  for (const keyword of costKeywords) {
    const matchIndex = colNames.findIndex((name) => name.includes(keyword));
    if (matchIndex !== -1) {
      columnRoles.value[columns.value[matchIndex].name] = ColumnRole.Cost;
      break;
    }
  }

  // Try to identify conversion columns
  const conversionKeywords = [
    'conversions',
    'impressions',
    'visits',
    'signups',
    'purchases',
    'revenue',
    'sale',
    'metric',
  ];
  for (const col of columns.value) {
    const name = col.name.toLowerCase();
    if (columnRoles.value[col.name] === ColumnRole.Ignore) {
      for (const keyword of conversionKeywords) {
        if (name.includes(keyword)) {
          columnRoles.value[col.name] = ColumnRole.Metric;
          break;
        }
      }
    }
  }

  // If no conversion columns were detected but we have numeric columns, use those
  const hasMetrics = Object.values(columnRoles.value).some((role) => role === ColumnRole.Metric);
  if (!hasMetrics) {
    columns.value.forEach((col) => {
      if (
        columnRoles.value[col.name] === ColumnRole.Ignore &&
        rawData.value.slice(0, 5).every((row) => typeof row[col.name] === 'number')
      ) {
        columnRoles.value[col.name] = ColumnRole.Metric;
      }
    });
  }

  await handleColumnRoleChange('', ColumnRole.Ignore);
}

async function fetchExternalData() {
  if (!externalSourceUrl.value) {
    showError('Please enter a valid Google Sheets URL');
    return;
  }

  loadingExternalData.value = true;
  try {
    const res = await getApiUi(
      'datasources/preview',
      {
        url: externalSourceUrl.value,
      },
      'Fetching data from external source',
    );

    if (!res || !res.data || !Array.isArray(res.data) || res.data.length === 0) {
      return;
    }

    await processData(res.data as Record<string, unknown>[]);

    // Auto-set data source name from URL if not set
    if (!currentDataSource.value.name) {
      const url = new URL(externalSourceUrl.value);
      currentDataSource.value.name = url.pathname.split('/').pop() || 'External Data Source';
    }

    $q.notify({
      type: 'positive',
      message: 'Successfully loaded data from external source',
    });
  } catch (error) {
    assertIsError(error);
    showError('Failed to load external data: ' + error.message);
    rawData.value = [];
    columns.value = [];
  } finally {
    loadingExternalData.value = false;
  }
}

async function saveDataSource() {
  if (!isFormValid.value) {
    showError('Please complete all required fields');
    return;
  }

  saving.value = true;

  try {
    // Always update from rawData if it has content, for both new and existing data sources
    if (rawData.value.length > 0) {
      currentDataSource.value.data = dataSourcesStore.normalizeRawData(
        rawData.value,
        currentDataSource.value.columns,
      );
    } else if (isEditing.value && !rawData.value.length) {
      currentDataSource.value.data = dataSourcesStore.normalizeRawData(
        [],
        currentDataSource.value.columns,
      );
    }

    const res = await dataSourcesStore.saveDataSource(currentDataSource.value);
    if (!res) return;

    emit('saved', res || currentDataSource.value);
  } catch (error) {
    showError('Failed to save data source: ' + error);
  } finally {
    saving.value = false;
  }
}
const scrollToRow = async (rowIndex: number) => {
  if (!dataPreviewTable.value) {
    return;
  }
  highlightedRow.value = rowIndex;
  const { rowsPerPage, sortBy, descending } = dataPreviewTable.value.pagination;

  // Use processedRawData to get all rows, not just the visible ones
  const allRows = processedRawData.value;
  const rowIndexInAllData = allRows.findIndex((row) => row.__index === rowIndex);

  if (rowIndexInAllData !== -1) {
    const page = Math.ceil((rowIndexInAllData + 1) / rowsPerPage);
    dataPreviewTable.value.setPagination({
      page,
      rowsPerPage,
      sortBy,
      descending,
    });

    await nextTick();

    const tableBody = dataPreviewTable.value.$el.querySelector('.q-table tbody');
    if (tableBody) {
      // Find the row within the newly rendered page
      const rowsOnPage = dataPreviewTable.value.computedRows;
      const rowIndexOnPage = rowsOnPage.findIndex((row) => row.__index === rowIndex);
      if (rowIndexOnPage !== -1) {
        const rowElement = tableBody.children[rowIndexOnPage];
        if (rowElement) {
          rowElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
          setTimeout(() => {
            highlightedRow.value = null;
          }, 3000); // Remove highlight after animation
        }
      }
    }
  }
};

function onCancel() {
  emit('canceled');
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function handleColumnRoleChange(columnName: string, newRole: any) {
  const roleValue = typeof newRole === 'object' && newRole !== null ? newRole.value : newRole;

  if (columnName) {
    columnRoles.value[columnName] = roleValue;
  }

  if (
    roleValue === ColumnRole.Geo ||
    roleValue === ColumnRole.Date ||
    roleValue === ColumnRole.Cost
  ) {
    Object.keys(columnRoles.value).forEach((col) => {
      if (col !== columnName && columnRoles.value[col] === roleValue) {
        columnRoles.value[col] = ColumnRole.Ignore;
      }
    });
  }

  currentDataSource.value.columns.geoColumn = '';
  currentDataSource.value.columns.dateColumn = '';
  currentDataSource.value.columns.metricColumns = [];
  currentDataSource.value.columns.costColumn = undefined;

  Object.entries(columnRoles.value).forEach(([colName, role]) => {
    switch (role) {
      case ColumnRole.Geo:
        currentDataSource.value.columns.geoColumn = colName;
        break;
      case ColumnRole.Date:
        currentDataSource.value.columns.dateColumn = colName;
        break;
      case ColumnRole.Metric:
        currentDataSource.value.columns.metricColumns.push(colName);
        break;
      case ColumnRole.Cost:
        currentDataSource.value.columns.costColumn = colName;
        break;
    }
  });
  updateDataStats();
  await triggerValidation();
}

function updateDataStats() {
  // Extract unique geo units
  const geoColumn = currentDataSource.value.columns.geoColumn;
  if (geoColumn) {
    const geoSet = new Set<string>();
    rawData.value.forEach((row) => {
      if (row[geoColumn] !== undefined && row[geoColumn] !== null) {
        geoSet.add(String(row[geoColumn]));
      }
    });
    geoUnits.value = Array.from(geoSet).sort();
  }

  // Extract unique dates
  const dateColumn = currentDataSource.value.columns.dateColumn;
  if (dateColumn) {
    const dateSet = new Set<string>();
    rawData.value.forEach((row) => {
      if (row[dateColumn] !== undefined && row[dateColumn] !== null) {
        dateSet.add(String(row[dateColumn]));
      }
    });
    uniqueDates.value = Array.from(dateSet).sort();
  }
}

function openFixDialog() {
  fixSelection.value = {
    removeMissingCritical: availableFixTypes.value.removeMissingCritical,
    fillMissingMetrics: availableFixTypes.value.fillMissingMetrics,
    fillDateGaps: availableFixTypes.value.fillDateGaps,
    deduplicateRows: availableFixTypes.value.deduplicateRows,
  };
  showFixSelectionDialog.value = true;
}

function applySelectedFixes() {
  showFixSelectionDialog.value = false;
  executeFixes(fixSelection.value);
}

function executeFixes(selection: FixSelection) {
  if (!rawData.value.length) return;

  isFixingIssues.value = true;
  let rowsRemovedCount = 0;
  let metricsFilledCount = 0;
  let dateGapsFilledCount = 0;
  let deduplicatedRowsCount = 0;

  const { geoColumn, dateColumn, metricColumns, costColumn } = currentDataSource.value.columns;

  let currentProcessedData: Record<string, unknown>[] = [...rawData.value];

  // 1. Remove rows with missing geo/date values
  if (selection.removeMissingCritical) {
    const tempData = currentProcessedData
      .map((row) => {
        let rowIsValid = true;
        if (geoColumn && !row[geoColumn]) {
          rowIsValid = false;
        }
        if (dateColumn && !row[dateColumn]) {
          rowIsValid = false;
        }
        if (costColumn && !row[costColumn]) {
          rowIsValid = false;
        }
        return rowIsValid ? row : null;
      })
      .filter((row): row is Record<string, unknown> => row !== null);
    rowsRemovedCount = currentProcessedData.length - tempData.length;
    currentProcessedData = tempData;
  }

  // 2. Fill missing metric values
  if (selection.fillMissingMetrics) {
    currentProcessedData = currentProcessedData.map((row) => {
      const newRow = { ...row };
      if (metricColumns.length > 0) {
        metricColumns.forEach((metric) => {
          if (!newRow[metric] && newRow[metric] !== 0) {
            newRow[metric] = 0;
            metricsFilledCount++;
          }
        });
      }
      if (costColumn) {
        if (!newRow[costColumn] && newRow[costColumn] !== 0) {
          newRow[costColumn] = 0;
          metricsFilledCount++;
        }
      }
      return newRow;
    });
  }

  // 3. Fill date gaps
  if (selection.fillDateGaps && dateColumn && currentProcessedData.length > 0) {
    const newRowsToAdd: Record<string, unknown>[] = [];
    const dataByGeo: Record<string, Record<string, unknown>[]> = {};

    if (geoColumn) {
      currentProcessedData.forEach((row) => {
        const geoVal = row[geoColumn];
        let geo: string;
        if (geoVal === null || geoVal === undefined) {
          geo = '__NULL_GEO__';
        } else if (
          typeof geoVal === 'string' ||
          typeof geoVal === 'number' ||
          typeof geoVal === 'boolean'
        ) {
          geo = String(geoVal);
        } else {
          geo = JSON.stringify(geoVal);
        }
        if (!dataByGeo[geo]) dataByGeo[geo] = [];
        dataByGeo[geo].push(row);
      });
    } else {
      dataByGeo['__NO_GEO__'] = [...currentProcessedData];
    }

    Object.keys(dataByGeo).forEach((geoKey) => {
      const geoGroupData = dataByGeo[geoKey].sort((a, b) => {
        const dateA = a[dateColumn];
        const dateB = b[dateColumn];
        const timeA =
          typeof dateA === 'string' || typeof dateA === 'number' || dateA instanceof Date
            ? new Date(dateA).getTime()
            : 0;
        const timeB =
          typeof dateB === 'string' || typeof dateB === 'number' || dateB instanceof Date
            ? new Date(dateB).getTime()
            : 0;
        return timeA - timeB;
      });
      if (geoGroupData.length === 0) return;

      const firstDateVal = geoGroupData[0][dateColumn];
      const lastDateVal = geoGroupData[geoGroupData.length - 1][dateColumn];

      if (
        !(
          typeof firstDateVal === 'string' ||
          typeof firstDateVal === 'number' ||
          firstDateVal instanceof Date
        ) ||
        !(
          typeof lastDateVal === 'string' ||
          typeof lastDateVal === 'number' ||
          lastDateVal instanceof Date
        )
      ) {
        return;
      }

      const minDate = new Date(firstDateVal);
      const maxDate = new Date(lastDateVal);
      const existingDates = new Set(
        geoGroupData
          .map((row) => {
            const dateVal = row[dateColumn];
            return typeof dateVal === 'string' ||
              typeof dateVal === 'number' ||
              dateVal instanceof Date
              ? new Date(dateVal).toISOString().split('T')[0]
              : '';
          })
          .filter((dateStr) => dateStr !== ''),
      );

      for (let d = new Date(minDate); d <= maxDate; d.setDate(d.getDate() + 1)) {
        const currentDateStr = d.toISOString().split('T')[0];
        if (!existingDates.has(currentDateStr)) {
          const newRow: Record<string, unknown> = { __index: -1 };
          if (geoColumn && geoKey !== '__NO_GEO__' && geoKey !== '__NULL_GEO__') {
            newRow[geoColumn] = geoKey;
          } else if (geoColumn) {
            newRow[geoColumn] = null;
          }
          newRow[dateColumn] = currentDateStr;
          metricColumns.forEach((metric) => {
            newRow[metric] = 0;
          });

          columns.value.forEach((colDef) => {
            if (!(colDef.name in newRow) && colDef.name !== '__index') {
              newRow[colDef.name] = undefined;
            }
          });

          newRowsToAdd.push(newRow);
          dateGapsFilledCount++;
        }
      }
    });

    if (newRowsToAdd.length > 0) {
      currentProcessedData.push(...newRowsToAdd);
      currentProcessedData.sort((a, b) => {
        if (geoColumn && a[geoColumn] !== b[geoColumn]) {
          const geoValA = a[geoColumn];
          const geoValB = b[geoColumn];
          let geoA_str: string;
          let geoB_str: string;

          if (geoValA === null || geoValA === undefined) geoA_str = '';
          else if (
            typeof geoValA === 'string' ||
            typeof geoValA === 'number' ||
            typeof geoValA === 'boolean'
          )
            geoA_str = String(geoValA);
          else geoA_str = JSON.stringify(geoValA);

          if (geoValB === null || geoValB === undefined) geoB_str = '';
          else if (
            typeof geoValB === 'string' ||
            typeof geoValB === 'number' ||
            typeof geoValB === 'boolean'
          )
            geoB_str = String(geoValB);
          else geoB_str = JSON.stringify(geoValB);

          return geoA_str.localeCompare(geoB_str);
        }
        const dateA = a[dateColumn];
        const dateB = b[dateColumn];
        const timeA =
          typeof dateA === 'string' || typeof dateA === 'number' || dateA instanceof Date
            ? new Date(dateA).getTime()
            : 0;
        const timeB =
          typeof dateB === 'string' || typeof dateB === 'number' || dateB instanceof Date
            ? new Date(dateB).getTime()
            : 0;
        return timeA - timeB;
      });
    }
  }

  // 4. Remove duplicate Geo/Date rows
  if (selection.deduplicateRows && dateColumn && geoColumn && currentProcessedData.length > 0) {
    const seenCombinations = new Set<string>();
    const uniqueRows: Record<string, unknown>[] = [];
    const originalLength = currentProcessedData.length;

    for (const row of currentProcessedData) {
      const dateVal = row[dateColumn];
      const geoVal = row[geoColumn];

      const formattedDate: string = new Date(dateVal as string | number | Date)
        .toISOString()
        .split('T')[0];

      let sGeoVal: string;
      if (geoVal === null || geoVal === undefined) {
        sGeoVal = 'N/A';
      } else if (
        typeof geoVal === 'string' ||
        typeof geoVal === 'number' ||
        typeof geoVal === 'boolean'
      ) {
        sGeoVal = String(geoVal);
      } else {
        sGeoVal = JSON.stringify(geoVal);
      }
      const combination = `${sGeoVal}|${formattedDate}`;

      if (!seenCombinations.has(combination)) {
        seenCombinations.add(combination);
        uniqueRows.push(row);
      }
    }
    deduplicatedRowsCount = originalLength - uniqueRows.length;
    currentProcessedData = uniqueRows;
  }

  rawData.value = currentProcessedData.map((row, index) => ({
    ...row,
    __index: index,
  }));

  let message = 'Fixes applied: ';
  const fixesApplied = [];
  if (rowsRemovedCount > 0)
    fixesApplied.push(`${rowsRemovedCount} row(s) with missing critical data removed`);
  if (metricsFilledCount > 0)
    fixesApplied.push(`${metricsFilledCount} missing metric value(s) filled`);
  if (dateGapsFilledCount > 0) fixesApplied.push(`${dateGapsFilledCount} date gap(s) filled`);
  if (deduplicatedRowsCount > 0)
    fixesApplied.push(`${deduplicatedRowsCount} duplicate row(s) removed`);

  if (fixesApplied.length === 0) {
    message = 'No fixes were applied based on selection or no relevant issues found.';
  } else {
    message += fixesApplied.join(', ') + '.';
  }

  $q.notify({
    type: fixesApplied.length > 0 ? 'positive' : 'info',
    message: message.trim(),
    position: 'top',
    timeout: 5000,
    actions: [{ icon: 'close', color: 'white', round: true, dense: true }],
  });

  isFixingIssues.value = false;
}
</script>

<style lang="scss" scoped>
@keyframes highlight-fade {
  from {
    background-color: #f0f8ff;
  }
  to {
    background-color: transparent;
  }
}

.highlighted-row {
  animation: highlight-fade 3s ease-out forwards;
}
</style>
