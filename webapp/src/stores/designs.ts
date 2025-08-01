import { defineStore } from 'pinia';
import { getApiUi, deleteApiUi, postApiUi } from 'src/boot/axios';
import { ref } from 'vue';
import type { SavedDesign } from 'src/components/models';

export const useDesignsStore = defineStore('designs', () => {
  const designs = ref<SavedDesign[]>([]);
  const loading = ref(false);
  const error = ref<string | null>(null);
  const isLoaded = ref(false);

  async function loadDesigns(reload = false) {
    if (isLoaded.value && !reload) {
      return designs.value;
    }

    loading.value = true;
    error.value = null;

    try {
      const response = await getApiUi<SavedDesign[]>('designs', {}, 'Loading designs');
      if (response && response.data) {
        designs.value = response.data;
        isLoaded.value = true;
      }
      return designs.value;
    } catch (err) {
      console.error('Failed to load designs from server:', err);
      error.value = (err as Error).message;
      throw err;
    } finally {
      loading.value = false;
    }
  }

  async function deleteDesign(design: SavedDesign) {
    loading.value = true;
    error.value = null;

    try {
      const response = await deleteApiUi(
        `designs/${design.design.design_id}`,
        'Deleting design...',
        `Are you sure you want to delete design ${design.design.design_id}?`,
      );
      if (response) {
        designs.value = designs.value.filter((d) => d.design.design_id !== design.design.design_id);
      }
      return response;
    } catch (err) {
      console.error('Failed to delete design:', err);
      error.value = (err as Error).message;
      throw err;
    } finally {
      loading.value = false;
    }
  }

  async function uploadDesign(design: SavedDesign) {
    try {
      await postApiUi(
        `designs/${design.design.design_id || Date.now()}.json`,
        design,
        'Uploading design to Cloud Storage',
      );
      // Add the design to the local store if it's not already there
      if (!designs.value.find((d) => d.design.design_id === design.design.design_id)) {
        designs.value.push(design);
      }
    } catch (err) {
      console.error('Failed to upload design:', err);
      error.value = (err as Error).message;
      throw err;
    }
  }

  function reset() {
    isLoaded.value = false;
    designs.value = [];
  }

  return {
    designs,
    loading,
    error,
    isLoaded,
    loadDesigns,
    deleteDesign,
    uploadDesign,
    reset,
  };
});
