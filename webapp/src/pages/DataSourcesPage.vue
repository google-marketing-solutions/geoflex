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
  <q-page padding>
    <div class="q-pb-md flex justify-between items-center">
      <div>
        <h1 class="text-h4 q-my-none">Data Sources</h1>
        <p class="text-body1 q-my-sm">Manage your data sources for geo-based marketing tests</p>
        <q-btn color="primary" icon="refresh" label="Reload" @click="loadDataSources(true)" />
      </div>
      <q-btn color="primary" icon="add" label="New Data Source" @click="openEditor()" />
    </div>

    <!-- Error state -->
    <q-banner v-if="error" class="bg-negative text-white q-mb-md">
      <template v-slot:avatar>
        <q-icon name="error" />
      </template>
      {{ error }}
      <template v-slot:action>
        <q-btn flat label="Retry" @click="loadDataSources(true)" />
      </template>
    </q-banner>

    <!-- Empty state -->
    <div v-if="!loading && !error && !datasources.length" class="text-center q-pa-xl">
      <q-icon name="source" size="6em" color="grey-5" />
      <div class="text-h5 q-mt-md">No Data Sources Yet</div>
      <p class="q-mb-lg">Create your first data source to get started with geo testing.</p>
      <q-btn color="primary" icon="add" label="Create Data Source" @click="openEditor()" />
    </div>

    <!-- Data Sources List -->
    <div v-if="!loading && datasources.length" class="q-mb-xl">
      <q-table
        :rows="datasources"
        :columns="columns"
        row-key="id"
        v-model:pagination="pagination"
        :filter="filter"
        :loading="loading"
      >
        <!-- Table top -->
        <template v-slot:top>
          <q-input
            v-model="filter"
            placeholder="Search data sources"
            dense
            debounce="300"
            class="q-mb-md"
          >
            <template v-slot:append>
              <q-icon name="search" />
            </template>
          </q-input>
        </template>

        <!-- Metrics column -->
        <template v-slot:body-cell-metrics="props">
          <q-td :props="props">
            <div class="ellipsis" style="max-width: 200px">
              {{ props.value.join(', ') }}
            </div>
          </q-td>
        </template>

        <!-- Status column -->
        <template v-slot:body-cell-status="props">
          <q-td :props="props">
            <q-badge :color="getStatus(props.row).color">
              {{ getStatus(props.row).label }}
            </q-badge>
          </q-td>
        </template>

        <!-- Actions column -->
        <template v-slot:body-cell-actions="props">
          <q-td :props="props" class="q-gutter-x-sm">
            <q-btn flat round dense icon="visibility" color="info" @click="openViewer(props.row)">
              <q-tooltip>View Data</q-tooltip>
            </q-btn>
            <q-btn flat round dense icon="edit" color="primary" @click="openEditor(props.row)">
              <q-tooltip>Edit</q-tooltip>
            </q-btn>
            <q-btn
              flat
              round
              dense
              icon="delete"
              color="negative"
              @click="confirmDelete(props.row)"
            >
              <q-tooltip>Delete</q-tooltip>
            </q-btn>
            <q-btn
              flat
              round
              dense
              icon="download"
              color="secondary"
              @click="downloadDataSource(props.row)"
            >
              <q-tooltip>Download CSV</q-tooltip>
            </q-btn>
          </q-td>
        </template>
      </q-table>
    </div>

    <!-- Editor Dialog -->
    <q-dialog v-model="showEditor" maximized persistent @hide="closeDialogs">
      <DataSourceEditor
        :data-source="selectedDataSource"
        @saved="onDataSourceSaved"
        @canceled="closeDialogs"
      />
    </q-dialog>

    <!-- Viewer Dialog -->
    <q-dialog v-model="showViewer" maximized @hide="closeDialogs">
      <DataSourceViewer v-if="selectedDataSource" :data-source="selectedDataSource" />
    </q-dialog>

    <!-- Delete Confirmation Dialog -->
    <q-dialog v-model="showDeleteConfirm" persistent>
      <q-card>
        <q-card-section class="row items-center">
          <q-avatar icon="warning" color="negative" text-color="white" />
          <span class="q-ml-sm">
            Are you sure you want to delete "{{ dataSourceToDelete?.name }}"?
          </span>
        </q-card-section>
        <q-card-section>
          This action cannot be undone. All associated data will be permanently removed.
        </q-card-section>
        <q-card-actions align="right">
          <q-btn flat label="Cancel" color="primary" v-close-popup />
          <q-btn
            label="Delete"
            color="negative"
            @click="deleteDataSource"
            :loading="deletingDataSource"
          />
        </q-card-actions>
      </q-card>
    </q-dialog>
  </q-page>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import type { QTableColumn } from 'quasar';
import { useQuasar } from 'quasar';
import type { DataSource } from 'stores/datasources';
import { useDataSourcesStore } from 'stores/datasources';
import { useUiSettingsStore } from 'stores/ui-settings';
import DataSourceViewer from 'components/DataSourceViewer.vue';
import DataSourceEditor from 'components/DataSourceEditor.vue';
import { formatDate } from 'src/helpers/utils';
import { getFile } from 'src/boot/axios';

const $q = useQuasar();
const dataSourcesStore = useDataSourcesStore();
const uiSettingsStore = useUiSettingsStore();
const router = useRouter();
const route = useRoute();

const COMPONENT_ID = 'DataSourcesPage';

// State
const filter = ref('');
const pagination = ref({
  sortBy: 'name',
  descending: false,
  page: 1,
  rowsPerPage: 10,
});
const showEditor = ref(false);
const showViewer = ref(false);
const showDeleteConfirm = ref(false);
const selectedDataSource = ref<DataSource | null>(null);
const dataSourceToDelete = ref<DataSource | null>(null);
const deletingDataSource = ref(false);

// Computed properties
const loading = computed(() => dataSourcesStore.loading);
const error = computed(() => dataSourcesStore.error);
const datasources = computed(() => dataSourcesStore.datasources);

// Table columns
const columns = [
  {
    name: 'name',
    label: 'Name',
    field: 'name',
    sortable: true,
    align: 'left',
  },
  {
    name: 'metrics',
    label: 'Metrics',
    field: (row: DataSource) => row.columns.metricColumns,
    align: 'left',
  },
  {
    name: 'status',
    label: 'Status',
    field: 'id',
    align: 'center',
  },
  {
    name: 'createdAt',
    label: 'Created',
    field: 'createdAt',
    sortable: true,
    align: 'left',
    format: (val: string) => formatDate(val),
  },
  {
    name: 'updatedAt',
    label: 'Updated',
    field: 'updatedAt',
    sortable: true,
    align: 'left',
    format: (val: string) => formatDate(val),
  },
  {
    name: 'actions',
    label: 'Actions',
    field: 'actions',
    align: 'center',
  },
] as QTableColumn[];

onMounted(() => {
  // Load saved UI settings
  const savedSettings = uiSettingsStore.getComponentSettings(COMPONENT_ID);
  if (savedSettings.filter) {
    filter.value = savedSettings.filter;
  }
  if (savedSettings.pagination) {
    pagination.value = savedSettings.pagination;
  }

  void loadDataSources(false); // Don't force reload on mount
});

// Watch for changes and save them to the store
watch(
  filter,
  (newFilter) => {
    uiSettingsStore.saveComponentSettings(COMPONENT_ID, { filter: newFilter });
  },
  { deep: true },
);

watch(
  pagination,
  (newPagination) => {
    uiSettingsStore.saveComponentSettings(COMPONENT_ID, { pagination: newPagination });
  },
  { deep: true },
);

// Load data sources
async function loadDataSources(reload = false) {
  try {
    await dataSourcesStore.loadDataSources(true, reload);
  } catch (err) {
    console.error('Failed to load data sources:', err);
  }
}

// Watcher is the single source of truth for dialog visibility
watch(
  () => route.params,
  async (params) => {
    const { id, action } = params;

    if (id && action) {
      await dataSourcesStore.loadDataSources(false, false); // Ensure data is loaded

      if (action === 'edit') {
        const isNew = id === 'new';
        const ds = isNew ? null : dataSourcesStore.getDataSourceById(id as string);

        if (!isNew && !ds) {
          $q.notify({ type: 'negative', message: `Datasource '${id as string}' not found.` });
          return closeDialogs();
        }
        selectedDataSource.value = ds;
        showViewer.value = false;
        showEditor.value = true;
      } else if (action === 'view') {
        const ds = dataSourcesStore.getDataSourceById(id as string);
        if (!ds) {
          $q.notify({ type: 'negative', message: `Datasource '${id as string}' not found.` });
          return closeDialogs();
        }
        selectedDataSource.value = ds;
        showEditor.value = false;
        showViewer.value = true;
      }
    } else {
      showEditor.value = false;
      showViewer.value = false;
    }
  },
  { immediate: true },
);

// Open the data source editor (for editing or creating)
function openEditor(dataSource?: DataSource) {
  const id = dataSource?.id || 'new';
  void router.push(`/datasources/${id}/edit`);
}

// Open the data source viewer
function openViewer(dataSource: DataSource) {
  void router.push(`/datasources/${dataSource.id}/view`);
}

// Download a data source
async function downloadDataSource(dataSource: DataSource) {
  await getFile(`datasources/${dataSource.id}/download`);
}

// Close all dialogs and navigate to the base URL
function closeDialogs() {
  if (route.params.id) {
    void router.push('/datasources');
  }
}

// Handle data source save event
function onDataSourceSaved(dataSource: DataSource) {
  // After saving, view the result
  void router.push(`/datasources/${dataSource.id}/view`);
}

// Confirm data source deletion
function confirmDelete(dataSource: DataSource) {
  dataSourceToDelete.value = dataSource;
  showDeleteConfirm.value = true;
}

// Delete a data source
async function deleteDataSource() {
  if (!dataSourceToDelete.value) return;

  deletingDataSource.value = true;

  try {
    await dataSourcesStore.deleteDataSource(dataSourceToDelete.value.id);

    $q.notify({
      type: 'positive',
      message: `Data source "${dataSourceToDelete.value.name}" deleted successfully`,
      position: 'top',
    });

    showDeleteConfirm.value = false;
  } catch {
    $q.notify({
      type: 'negative',
      message: 'Failed to delete data source',
      position: 'top',
    });
  } finally {
    deletingDataSource.value = false;
    dataSourceToDelete.value = null;
  }
}

// Get status badge for a data source
function getStatus(dataSource: DataSource) {
  if (!dataSource.data) {
    return { color: 'warning', label: 'Not loaded' };
  }

  if (dataSource.data && dataSource.data.rawRows?.length === 0) {
    return { color: 'warning', label: 'Empty' };
  }

  return { color: 'positive', label: 'Active' };
}
</script>
