<template>
  <q-page padding>
    <q-card class="configuration-card">
      <q-card-section>
        <div class="text-h5 q-mt-xs q-mb-lg">Application Configuration</div>

        <div>
          <!-- Master Spreadsheet ID -->
          <q-input
            v-model="masterSpreadsheetId"
            label="Master Spreadsheet ID *"
            filled
            :rules="[(val) => !!val || 'Spreadsheet ID is required']"
            :loading="loading"
            :disable="loading"
          >
            <template v-slot:prepend>
              <q-icon name="table_chart" />
            </template>
            <template v-slot:append v-if="masterSpreadsheetId">
              <q-btn
                round
                flat
                dense
                icon="open_in_new"
                type="button"
                @click="onOpenSpreadsheet"
                class="q-ml-sm"
              >
                <q-tooltip>Open in Google Sheets</q-tooltip>
              </q-btn>
            </template>
            <template v-slot:hint>
              The ID of your master Google Spreadsheet where datasource definitions are stored
            </template>
          </q-input>

          <!-- Action Buttons -->
          <div class="row justify-between q-mt-lg">
            <q-btn
              label="Reload"
              flat
              :disable="loading"
              @click="reload"
              icon="sync"
            />

            <div class="row q-gutter-sm">
              <q-btn
                label="Share With Me"
                color="secondary"
                icon="share"
                :loading="sharingLoading"
                :disable="loading || !masterSpreadsheetId"
                @click="shareSpreadsheet"
              />

              <q-btn
                label="Update"
                color="primary"
                icon="save"
                :disable="!hasChanges()"
                :loading="loading"
                @click="updateConfiguration"
              />

              <q-btn
                label="Recreate"
                color="warning"
                icon="create"
                :loading="loading"
                @click="createSpreadsheet"
              />
            </div>
          </div>
        </div>
      </q-card-section>
    </q-card>

    <!-- Instructions Card -->
    <q-card class="q-mt-lg">
      <q-card-section>
        <div class="text-h6">Setup Instructions</div>

        <ol class="q-pl-md">
          <li class="q-mb-sm">
            Master Spreadsheet is created automatically but you can attach an existing one.
          </li>
          <li class="q-mb-sm">
            If you're using an existing spreadsheet, make sure it has a sheet named "DataSources"
            with the correct headers: id, name, description, created_at, updated_at, source_link,
            columns.
          </li>
          <li class="q-mb-sm">
            After updating the configuration, click "Share With Me" to ensure you have access.
          </li>
          <li class="q-mb-sm">Go to the Data Sources page to start managing your data sources.</li>
        </ol>
      </q-card-section>
    </q-card>
  </q-page>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { useQuasar } from 'quasar';
import axios from 'axios';

// Quasar utilities
const $q = useQuasar();

// Form state
const masterSpreadsheetId = ref('');
const originalId = ref('');
const loading = ref(false);
const sharingLoading = ref(false);

// Load current configuration
onMounted(async () => {
  await loadConfiguration();
});

function onOpenSpreadsheet(event) {
  event.preventDefault();
  event.stopPropagation();
  const url = `https://docs.google.com/spreadsheets/d/${masterSpreadsheetId.value}/edit`;
  window.open(url, '_blank');
}

async function loadConfiguration() {
  loading.value = true;

  try {
    const response = await axios.get('/api/config');
    masterSpreadsheetId.value = response.data.spreadsheet_id;
    originalId.value = response.data.spreadsheet_id;
  } catch (error) {
    showError('Failed to load configuration');
    console.error('Error loading configuration:', error);
  } finally {
    loading.value = false;
  }
}

function hasChanges() {
  return originalId.value !== masterSpreadsheetId.value;
}

// Update configuration
async function updateConfiguration() {
  if (!masterSpreadsheetId.value) return;

  loading.value = true;

  try {
    await axios.put('/api/config', {
      spreadsheet_id: masterSpreadsheetId.value,
    });

    originalId.value = masterSpreadsheetId.value;

    $q.notify({
      type: 'positive',
      message: 'Configuration updated successfully',
      position: 'top',
    });
  } catch (error) {
    showError('Failed to update configuration');
    console.error('Error updating configuration:', error);
  } finally {
    loading.value = false;
  }
}

// Share spreadsheet with current user
async function shareSpreadsheet() {
  if (!masterSpreadsheetId.value) return;

  sharingLoading.value = true;

  try {
    const response = await axios.post('/api/config/share');

    $q.notify({
      type: 'positive',
      message: response.data.message || 'Spreadsheet shared successfully',
      position: 'top',
    });
  } catch (error) {
    showError('Failed to share spreadsheet');
    console.error('Error sharing spreadsheet:', error);
  } finally {
    sharingLoading.value = false;
  }
}

async function createSpreadsheet() {
  loading.value = true;

  try {
    const response = await axios.post('/api/config/recreate');
    masterSpreadsheetId.value = response.data.spreadsheet_id;
    originalId.value = response.data.spreadsheet_id;

    $q.notify({
      type: 'positive',
      message: response.data.message || 'Spreadsheet recreated successfully',
      position: 'top',
    });
  } catch (error) {
    showError('Failed to create master spreadsheet');
    console.error('Error creating spreadsheet:', error);
  } finally {
    loading.value = false;
  }
}

// Reset form to original values
async function reload() {
  await loadConfiguration();
}

// Show error notification
function showError(message: string) {
  $q.notify({
    type: 'negative',
    message,
    position: 'top',
  });
}
</script>

<style scoped>
.configuration-card {
}
</style>
