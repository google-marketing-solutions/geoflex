<template>
  <q-layout view="hHh Lpr lFf">
    <q-header elevated>
      <q-toolbar>
        <q-btn flat dense round icon="menu" aria-label="Menu" @click="toggleLeftDrawer" />

        <q-toolbar-title class="flex items-center">
          <AppLogo :size="40" class="q-mr-sm" />
          <router-link to="/" style="text-decoration: none; color: white"
            >GeoFlex Testing Platform</router-link
          ></q-toolbar-title
        >

        <q-btn
          :icon="$q.dark.isActive ? 'light_mode' : 'dark_mode'"
          flat
          round
          @click="$q.dark.toggle()"
          aria-label="Toggle dark mode"
        />
      </q-toolbar>
    </q-header>

    <q-drawer v-model="leftDrawerOpen" show-if-above bordered>
      <SideMenu />
    </q-drawer>

    <q-page-container>
      <router-view />
    </q-page-container>

    <q-footer :class="$q.dark.isActive ? 'bg-grey-10 text-white' : 'bg-white text-dark'">
      <div class="text-body1 text-center q-ma-sm">
        &copy;&nbsp;Google gTech Ads, 2025. Built
        {{ formattedBuildTime }} (git#{{ GIT_HASH }}) (not an official Google product)
      </div>
    </q-footer>
  </q-layout>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import AppLogo from 'components/Logo.vue';
import SideMenu from 'components/SideMenu.vue';

defineOptions({
  name: 'MainLayout',
});

const leftDrawerOpen = ref(false);

function toggleLeftDrawer() {
  leftDrawerOpen.value = !leftDrawerOpen.value;
}

const BUILD_TIMESTAMP = process.env.BUILD_TIMESTAMP;
const GIT_HASH = process.env.GIT_HASH;

const formattedBuildTime = BUILD_TIMESTAMP ? new Date(BUILD_TIMESTAMP).toLocaleString() : '';
</script>
