<template>
  <q-page padding>
    <div class="q-pb-md">
      <div>
        <h1 class="text-h4 q-my-none">Saved Designs</h1>
        <p class="text-body1 q-my-sm">Manage your saved experiment designs</p>
        <q-btn color="primary" icon="refresh" label="Reload" @click="reloadDesigns" />
      </div>
    </div>
    <design-list
      class="q-mb-xl"
      :designs="designsStore.designs"
      :showAnalyze="true"
      :show-meta="true"
      :show-delete="true"
      :default-sort="sortBy"
      @update:sort-by="updateSortBy"
      @analyze="analyzeDesign"
      @delete="deleteDesign"
      @update="updateDesign"
    />
  </q-page>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import DesignList from 'src/components/DesignList.vue';
import type { SavedDesign } from 'src/components/models';
import { useRouter } from 'vue-router';
import { useDesignsStore } from 'src/stores/designs';
import { useUiSettingsStore } from 'src/stores/ui-settings';

const router = useRouter();
const designsStore = useDesignsStore();
const uiSettingsStore = useUiSettingsStore();

const COMPONENT_ID = 'DesignsPage';
type SortByType = 'mde' | 'duration' | 'recency';
const sortBy = ref<SortByType>('recency');

async function reloadDesigns() {
  await designsStore.loadDesigns(true);
}

async function deleteDesign(design: SavedDesign) {
  await designsStore.deleteDesign(design);
}

async function updateDesign(design: SavedDesign, newValues: Partial<SavedDesign>) {
  await designsStore.updateDesign(design, newValues);
}

function analyzeDesign(design: SavedDesign) {
  router
    .push({
      name: 'analysis',
      params: {
        designId: design.design.design_id,
      },
    })
    .catch((err) => {
      console.error('Router push failed:', err);
    });
}

function updateSortBy(newSortBy: SortByType) {
  sortBy.value = newSortBy;
  uiSettingsStore.saveComponentSettings(COMPONENT_ID, { sortBy: newSortBy });
}

onMounted(async () => {
  const savedSettings = uiSettingsStore.getComponentSettings(COMPONENT_ID);
  if (savedSettings.sortBy) {
    sortBy.value = savedSettings.sortBy as SortByType;
  }
  await designsStore.loadDesigns();
});
</script>
