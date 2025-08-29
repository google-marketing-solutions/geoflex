<template>
  <div>
    <apexchart type="rangeBar" height="200" :options="chartOptions" :series="series"></apexchart>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { useQuasar } from 'quasar';
import type { ApexOptions } from 'apexcharts';

const $q = useQuasar();

const props = defineProps({
  design: {
    type: Object,
    required: true,
  },
  dataSource: {
    type: Object,
    required: true,
  },
  experimentStartDate: {
    type: String,
    required: true,
  },
  experimentEndDate: {
    type: String,
    default: null,
  },
});

const series = computed(() => {
  if (
    !props.design ||
    !props.dataSource ||
    !props.dataSource.data ||
    !props.dataSource.data.uniqueDates ||
    props.dataSource.data.uniqueDates.length < 1 ||
    !props.experimentStartDate
  ) {
    return [];
  }

  // Design data
  const designStartDate = new Date(props.design.start_date);
  const designEndDate = new Date(props.design.end_date);
  if (isNaN(designStartDate.getTime()) || isNaN(designEndDate.getTime())) {
    return [];
  }

  // Datasource full range
  const dsStartDate = new Date(props.dataSource.data.uniqueDates[0]);
  const dsEndDate = new Date(
    props.dataSource.data.uniqueDates[props.dataSource.data.uniqueDates.length - 1],
  );
  if (isNaN(dsStartDate.getTime()) || isNaN(dsEndDate.getTime())) {
    return [];
  }

  // Experiment data
  const experimentStartDate = new Date(props.experimentStartDate);
  let experimentEndDate;
  if (props.experimentEndDate) {
    experimentEndDate = new Date(props.experimentEndDate);
  } else {
    experimentEndDate = new Date(
      experimentStartDate.getTime() + props.design.design.runtime_weeks * 7 * 24 * 60 * 60 * 1000,
    );
  }
  console.log('Experiment period: ', experimentStartDate, experimentEndDate);

  if (
    isNaN(experimentStartDate.getTime()) ||
    !experimentEndDate ||
    isNaN(experimentEndDate.getTime())
  ) {
    return [];
  }
  const experimentData = {
    data: [
      {
        x: 'Design Period',
        y: [designStartDate.getTime(), designEndDate.getTime()],
        fillColor: '#008FFB',
      },
      {
        x: 'Analysis Data Period',
        y: [dsStartDate.getTime(), dsEndDate.getTime()],
        fillColor: '#00E396',
      },
      {
        x: 'Experiment Period',
        y: [experimentStartDate.getTime(), experimentEndDate.getTime()],
        fillColor: '#FEB019',
      },
    ],
  };

  return [experimentData];
});

const chartOptions = ref<ApexOptions>({
  title: { text: 'Experiment timeline', margin: 15 },
  chart: {
    height: 200,
    type: 'rangeBar',
    zoom: {
      enabled: false,
    },
  },
  plotOptions: {
    bar: {
      horizontal: true,
      distributed: true,
      barHeight: '70%',
    },
  },
  yaxis: { labels: { show: false } },
  xaxis: {
    type: 'datetime',
    labels: {
      datetimeFormatter: {
        year: 'yyyy',
        month: 'MM.yy',
        day: 'dd.MM.yyyy',
      },
      style: {
        colors: $q.dark.isActive ? '#FFFFFF' : '#333333',
      },
    },
  },
  dataLabels: {
    enabled: true,
    formatter: function (val, opts) {
      const label = opts.w.globals.labels[opts.dataPointIndex];
      const start = new Date(val[0]);
      const end = new Date(val[1]);
      const diffTime = Math.abs(end.getTime() - start.getTime());
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
      const diffWeeks = Math.round(diffDays / 7);
      return `${label}: ${diffDays} ${diffDays > 1 ? ' days' : ' day'} (${diffWeeks}w)`;
    },
  },
  theme: {
    mode: $q.dark.isActive ? 'dark' : 'light',
  },
});
</script>
